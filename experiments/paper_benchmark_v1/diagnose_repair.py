"""One-hour inference-only factorial diagnosis of S2 on frozen dev sets."""
import argparse
import json
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PrivatePaperDataset
from .diagnose_phase1 import DiagnosticData, digest, regional
from .model import CMXPaper
from .noise_model import NoiseCMX
from .run import evaluate
from .run_repair_window import load_frontend
from .train_reliability import read, save

CONDITIONS = [('clean', 'medium'), ('photon_proxy', 'heavy'),
              ('joint', 'heavy'), ('depth_noise', 'heavy')]
MODES = ('R0_off', 'R0_on', 'S2_off', 'S2_on')


class FactorModel(NoiseCMX):
    def __init__(self, enabled, **kwargs):
        super().__init__('S2', **kwargs)
        self.enabled = enabled

    def forward(self, inputs):
        if self.enabled:
            return super().forward(inputs)
        return CMXPaper.forward(self, inputs)


def segmentation_state(state):
    return {k: v for k, v in state.items() if k.startswith(('backbone.', 'decode_head.'))}


def build_model(segmentation, frontend, enabled, classes=40, size=640, decoder_dim=256):
    model = FactorModel(enabled, num_classes=classes, image_size=size, decoder_dim=decoder_dim)
    model.load_baseline(segmentation_state(segmentation))
    load_frontend(model, frontend)
    return model.eval()


def effects(values):
    # Factorial contrasts in percentage points, not causal proof of training dynamics.
    return dict(frontend_on_R0_pp=100*(values['R0_on']-values['R0_off']),
                frontend_on_S2_pp=100*(values['S2_on']-values['S2_off']),
                segmentation_change_off_pp=100*(values['S2_off']-values['R0_off']),
                full_change_pp=100*(values['S2_on']-values['R0_off']),
                interaction_pp=100*((values['S2_on']-values['S2_off'])-
                                    (values['R0_on']-values['R0_off'])))


def update(args, **values):
    save(args.output_dir/'status.json', dict(updated_unix=time.time(), **values))


def preflight(args):
    prior = read(args.repair_dir/'protocol.json')
    files = {'R0': args.reference_dir/'mixed/last.pth',
             'S2': args.repair_dir/'S2_seed42/last.pth',
             'private_checkpoint': args.private_checkpoint}
    files.update({f'nyu_{s}': args.nyuv2_dir/'manifests'/f'{s}.jsonl' for s in ('train', 'dev')})
    for key in prior['private_sha256']:
        files['private/'+key] = (args.private_checkpoint if key=='checkpoint' else
                                 args.private_dir/Path(key.replace('\\', '/')))
    fingerprints = {key: digest(path) for key, path in files.items()}
    if fingerprints['S2'] != read(args.repair_dir/'S2_seed42/result.json')['checkpoint_sha256']:
        raise ValueError('S2 checkpoint differs from completed result')
    if fingerprints['R0'] != prior['references']['R0_seed42']['checkpoint']:
        raise ValueError('R0 checkpoint differs from repair-window reference')
    for s in ('train', 'dev'):
        if fingerprints[f'nyu_{s}'] != prior['manifests'][s]:
            raise ValueError('NYUv2 manifest changed')
    for key, fingerprint in prior['private_sha256'].items():
        if fingerprints['private/'+key] != fingerprint:
            raise ValueError(f'Private input changed: {key}')
    package = Path(__file__).parent
    code = sorted(package.glob('*.py')) + sorted((package.parent/'cmx_initial_transfer/official').glob('*.py'))
    references = [args.repair_dir/'private_dev_transfer.json']
    for family, severity in CONDITIONS:
        name = f'{family}_{severity}.json'
        references += [args.reference_dir/'mixed/curves'/name, args.repair_dir/'S2_seed42/curves'/name]
    protocol = dict(version=1, conditions=CONDITIONS, modes=MODES, fingerprints=fingerprints,
                    code={str(p.relative_to(package.parent)): digest(p) for p in code},
                    reference_hashes=[digest(p) for p in references],
                    scope='seed42 dev only; no training; crossed weights may have co-adaptation mismatch')
    # JSON normalization makes tuples stable across resume.
    protocol = json.loads(json.dumps(protocol))
    path = args.output_dir/'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Frozen diagnostic protocol changed; use a new output directory')
    save(path, protocol)


@torch.no_grad()
def private_audit(model, dataset, device):
    rows = []
    for item in dataset:
        x = item['inputs'][None].to(device)
        out = model.restore(x)
        delta = (out['restored_intensity']-x[:, :1]).abs()[0, 0].cpu()
        labels = item['labels']
        row = dict(sample_id=item['sample_id'], gate=float(out['noise_gate'].mean()),
                   correction_mae=float(delta.mean()))
        for key, mask in [('foreground', labels==1), ('background', labels==0),
                          ('invalid_depth', item['inputs'][2]==0)]:
            row[key+'_correction_mae'] = float(delta[mask].mean()) if bool(mask.any()) else None
        rows.append(row)
    gates = np.array([r['gate'] for r in rows])
    return dict(per_frame=rows, gate_mean=float(gates.mean()),
                gate_quantiles=dict(zip(['p0', 'p25', 'p50', 'p75', 'p100'],
                                       np.quantile(gates, [0, .25, .5, .75, 1]).tolist())),
                gate_fraction_above_half=float((gates>.5).mean()),
                correction_mae_frame_mean=float(np.mean([r['correction_mae'] for r in rows])),
                interpretation='No private true-noise labels: gate activation is not calibrated noise severity')


def execute(args):
    started = time.time()
    update(args, phase='preflight')
    preflight(args)
    r0 = torch.load(args.reference_dir/'mixed/last.pth', map_location='cpu', weights_only=False)
    s2 = torch.load(args.repair_dir/'S2_seed42/last.pth', map_location='cpu', weights_only=False)
    if r0['epoch'] != 20 or s2['epoch'] != 20 or s2['group'] != 'S2_seed42':
        raise ValueError('Expected completed epoch20 checkpoints')
    # Retain model tensors only; release optimizer states before allocating GPU model.
    r0, s2 = r0['model'], s2['model']
    scores = {f'{f}_{s}': {} for f, s in CONDITIONS}
    completed = 0
    for mode in MODES:
        model = build_model(r0 if mode.startswith('R0') else s2, s2, mode.endswith('_on')).to(args.device)
        for family, severity in CONDITIONS:
            condition = f'{family}_{severity}'
            update(args, phase='public_diagnostic', completed=completed, total=18, mode=mode, condition=condition)
            path = args.output_dir/'curves'/f'{mode}_{condition}.json'
            row = read(path) if path.exists() else regional(model, DiagnosticData(args.nyuv2_dir, family, severity), args.device)
            refdir = args.reference_dir/'mixed' if mode.startswith('R0') else args.repair_dir/'S2_seed42'
            ref = read(refdir/'curves'/f'{condition}.json')
            if row['degradation_audit'] != ref['degradation_audit']:
                raise ValueError('Diagnostic samples/corruption budget changed')
            score = row['regions']['all']['miou']
            if mode in ('R0_off', 'S2_on') and abs(score-ref['regions']['all']['miou']) > 1e-5:
                raise ValueError(f'Checkpoint does not reproduce saved metrics: {mode}/{condition}')
            save(path, row)
            scores[condition][mode] = score
            completed += 1
            print(f'DIAG_COMPLETE {completed}/18 mode={mode} condition={condition} miou={score:.6f}', flush=True)
        del model
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    comparisons = {key: effects(value) for key, value in scores.items()}
    save(args.output_dir/'public_summary.json', dict(miou=scores, contrasts_pp=comparisons,
         caveat='Off contrasts include parameters and buffers; swapping alone cannot identify their separate causes'))
    del r0
    private = torch.load(args.private_checkpoint, map_location='cpu', weights_only=False)
    if any(private['args'].get(k)!=v for k, v in dict(dataset='private', model='CMX_B2', seed=42, stage='tune').items()):
        raise ValueError('Wrong private source checkpoint')
    dim = private['args']['decoder_dim']
    private = private['model']
    dataset = PrivatePaperDataset(args.private_dir, args.private_dir/'paper_split_v1/dev_indices.npy')
    previous = read(args.repair_dir/'private_dev_transfer.json')['metrics']
    private_results = {}
    for enabled in (False, True):
        name = 'S2' if enabled else 'baseline'
        update(args, phase='private_diagnostic', completed=completed, total=18, mode=name)
        path = args.output_dir/f'private_{name}.json'
        model = build_model(private, s2, enabled, 2, 128, dim).to(args.device)
        if path.exists():
            row = read(path)
        else:
            row = dict(metrics=evaluate(model, DataLoader(dataset, batch_size=1, num_workers=0,
                                                        pin_memory=False), args.device, 2))
            if enabled:
                row['frontend_audit'] = private_audit(model, dataset, args.device)
        if abs(row['metrics']['foreground_iou']-previous[name]['foreground_iou']) > 1e-5:
            raise ValueError(f'Private metrics not reproduced: {name}')
        save(path, row)
        private_results[name] = row
        completed += 1
        print(f'DIAG_COMPLETE {completed}/18 private={name} foreground_iou={row["metrics"]["foreground_iou"]:.6f}', flush=True)
        del model
    save(args.output_dir/'summary.json', dict(public=comparisons, private=private_results,
         next_step='Inspect segmentation-change and frontend contrasts before choosing frozen-backbone training; no automatic success claim'))
    for condition, contrasts in comparisons.items():
        print(f'DIAG_CONTRAST {condition} {json.dumps(contrasts)}', flush=True)
    save(args.output_dir/'complete.json', dict(evaluations=completed, training_runs=0,
                                             elapsed_minutes=(time.time()-started)/60))
    (args.output_dir/'failure.json').unlink(missing_ok=True)
    update(args, phase='complete', completed=completed, total=18)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(reference_dir='data/paper_benchmark_runs/mixed_control_v1',
                    repair_dir='data/paper_benchmark_runs/repair_window_v1',
                    output_dir='data/paper_benchmark_runs/repair_diagnosis_v1',
                    nyuv2_dir='data/public_semseg/nyuv2/processed', private_dir='data/new_data/merged',
                    private_checkpoint='data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B2/seed_42/best.pth')
    for name, value in defaults.items():
        parser.add_argument('--'+name.replace('_', '-'), type=Path, default=Path(value))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir/'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname()), handle)
    try:
        execute(args)
    except BaseException:
        error = traceback.format_exc()
        save(args.output_dir/'failure.json', dict(error=error))
        update(args, phase='failed', error=error)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
