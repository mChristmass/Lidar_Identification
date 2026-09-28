"""Eight-hour clean-preserving restoration screen; frozen development sets only."""
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

from . import run_noise_window as noise
from .data import PrivatePaperDataset
from .diagnose_phase1 import DiagnosticData, digest
from .model import CMXPaper
from .noise_model import NoiseCMX
from .run import evaluate
from .run_fusion_window import train_job
from .train_reliability import read, save


def choose(results):
    eligible = []
    for result in results:
        gains = {r['condition']: r['gain_pp'] for r in result['scores']}
        noise_gain = np.mean([v for k, v in gains.items()
                              if k.startswith(('photon_proxy_', 'joint_'))])
        if (gains['clean_medium'] >= -.15 and
                result['corruption_mean_gain_pp'] >= .25 and noise_gain >= .8):
            eligible.append(result)
    return max(eligible, key=lambda r: r['corruption_mean_gain_pp']) if eligible else None


def checked_reference(path):
    result = read(path / 'result.json')
    fingerprint = digest(path / 'last.pth')
    legacy = 'checkpoint_sha256' not in result and 'protocol' in result
    if not legacy and result['checkpoint_sha256'] != fingerprint:
        raise ValueError(f'Reference checkpoint changed: {path}')
    refs = noise.references(path / 'curves')
    if any((row.get('initialization_sha256') != result['protocol']['checkpoint_sha256']
            or row.get('regime') != 'mixed') if legacy else
           row.get('checkpoint_sha256') != fingerprint for row in refs.values()):
        raise ValueError(f'Reference provenance mismatch: {path}')
    if any(row['epoch'] != 20
           for row in refs.values()):
        raise ValueError(f'Reference curves do not match epoch20: {path}')
    if len(result['scores']) != 22 or any(
            r['miou'] != refs[r['condition']]['regions']['all']['miou'] for r in result['scores']):
        raise ValueError(f'Reference summary/curves mismatch: {path}')
    result['clean_miou'] = refs['clean_medium']['regions']['all']['miou']
    result['corruption_mean_miou'] = float(np.mean([
        r['miou'] for r in result['scores'] if r['condition'] != 'clean_medium']))
    return result, refs


def preflight(args):
    old = read(args.reference_dir / 'protocol.json')
    manifest = args.nyuv2_dir / 'manifests'
    hashes = {s: digest(manifest / f'{s}.jsonl') for s in ('train', 'dev')}
    if any(old[f'{s}_manifest_sha256'] != hashes[s] for s in hashes):
        raise ValueError('Frozen NYUv2 manifests changed')
    sources = {str(s): digest(noise.source_path(args, s)) for s in (42, 777, 2025)}
    if sources['42'] != old['checkpoint_sha256']:
        raise ValueError('Clean seed42 source changed')
    prior = read(args.noise_dir / 'protocol.json')
    if (prior['manifest_sha256'] != hashes or
            any(prior['checkpoint_sha256'][s] != sources[s] for s in ('42', '777'))):
        raise ValueError('Noise-window source or manifests differ')
    references = {}
    for name, value in prior['reference_sha256'].items():
        if digest(args.reference_dir / 'mixed/curves' / f'{name}.json') != value:
            raise ValueError('Legacy R0 curves changed since noise window')
    for name, path in reference_paths(args).items():
        result, curves = checked_reference(path)
        references[name] = dict(checkpoint=digest(path / 'last.pth'),
                               result=digest(path / 'result.json'),
                               curves={k: digest(path / 'curves' / f'{k}.json') for k in curves})
    split = args.private_dir / 'paper_split_v1'
    dev = np.load(split / 'dev_indices.npy')
    train = np.load(split / 'train_indices.npy')
    if (len(dev) != 137 or len(np.unique(dev)) != len(dev) or
            np.intersect1d(dev, train).size or len(train) != 970):
        raise ValueError('Unexpected private frozen development split')
    private_files = [args.private_checkpoint, split / 'dev_indices.npy',
                     split / 'train_indices.npy', args.private_dir / 'intensity.npy',
                     args.private_dir / 'depth.npy']
    private_files += [args.private_dir / 'label' / f'{int(i)}.png' for i in dev]
    package = Path(__file__).parent
    code = sorted(package.glob('*.py')) + sorted(
        (package.parent / 'cmx_initial_transfer/official').glob('*.py'))
    protocol = dict(version=1, budget_hours=args.budget_hours, epochs=20,
                    source_sha256=sources, manifests=hashes, references=references,
                    private_sha256={str(p.relative_to(args.private_dir)) if p.is_relative_to(args.private_dir)
                                    else 'checkpoint': digest(p) for p in private_files},
                    code_sha256={str(p.relative_to(package.parent)): digest(p) for p in code},
                    recipe='D1 unchanged; S1 inactive identity weight1 (D1 .05); S2 adds image gate',
                    loss='CE+.2 KL+5 reconstruction+.1 gateBCE(S2 only); gate negatives weight .2',
                    selection='fixed epoch20, no checkpoint search',
                    gate='clean >=-.15pp AND 21-corrupt mean >=.25pp AND 6-noise mean >=.8pp',
                    confirmation='winner seed777; R0,D1,winner seed2025; >=270min initially, >=65min per new job',
                    scope='development only; private frontend zero-shot diagnostic, not private training')
    path = args.output_dir / 'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Protocol changed. Use a new output directory; do not overwrite old results.')
    save(path, protocol)
    return protocol


def reference_paths(args):
    return {'R0_seed42': args.reference_dir / 'mixed',
            'R0_seed777': args.noise_dir / 'R0_seed777',
            'D1_seed42': args.noise_dir / 'D1_seed42'}


@torch.no_grad()
def frontend_audit(model, args):
    model.eval()
    results = {}
    for family, severity in [('clean', 'medium')] + [('photon_proxy', s) for s in ('light', 'medium', 'heavy')]:
        rows = []
        for item in DiagnosticData(args.nyuv2_dir, family, severity):
            x = item['inputs'][None].to(args.device)
            out = model.restore(x)
            rows.append([out['noise_gate'].mean().item(),
                         (out['restored_intensity']-x[:, :1]).abs().mean().item()])
        values = np.asarray(rows)
        results[f'{family}_{severity}'] = dict(gate_mean=float(values[:, 0].mean()),
                                               correction_mae=float(values[:, 1].mean()))
    return results


def load_frontend(model, state):
    frontend = {k: v for k, v in state.items() if k.startswith(('restorer.', 'noise_gate.'))}
    expected = {k for k in model.state_dict() if k.startswith(('restorer.', 'noise_gate.'))}
    if set(frontend) != expected:
        raise ValueError('Frontend parameter keys differ')
    model.load_state_dict(frontend, strict=False)


@torch.no_grad()
def private_diagnostic(args):
    source = torch.load(args.private_checkpoint, map_location='cpu', weights_only=False)
    if any(source['args'].get(k) != v for k, v in
           dict(dataset='private', model='CMX_B2', seed=42, stage='tune').items()):
        raise ValueError('Incorrect private B2 seed42 checkpoint')
    dataset = PrivatePaperDataset(args.private_dir, args.private_dir / 'paper_split_v1/dev_indices.npy')
    loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)
    outputs = {}
    for method in ('baseline', 'D1', 'S1', 'S2'):
        noise.update(args, phase='private_dev_diagnostic', diagnostic_model=method)
        if method == 'baseline':
            model = CMXPaper('CMX_B2', 2, 128, source['args']['decoder_dim'])
            model.load_state_dict(source['model'], strict=True)
        else:
            model = NoiseCMX(method, num_classes=2, image_size=128,
                             decoder_dim=source['args']['decoder_dim'])
            model.load_baseline(source['model'])
            path = (args.noise_dir if method == 'D1' else args.output_dir) / f'{method}_seed42/last.pth'
            state = torch.load(path, map_location='cpu', weights_only=False)
            load_frontend(model, state['model'])
            del state
        model.to(args.device).eval()
        outputs[method] = evaluate(model, loader, args.device, 2)
        del model
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
        save(args.output_dir / 'private_dev_transfer.json',
             dict(scope='zero-shot public frontend on fixed private B2; development only', metrics=outputs))
        print(f'PRIVATE_DEV model={method} foreground_iou={outputs[method]["foreground_iou"]:.6f}', flush=True)
    return outputs


def summarize(args, results):
    aggregate = {}
    for method in sorted({r['method'] for r in results}):
        rows = [r for r in results if r['method'] == method]
        aggregate[method] = {'seeds': [r['seed'] for r in rows]}
        for key in ('clean_miou', 'corruption_mean_miou'):
            values = [r[key] for r in rows]
            aggregate[method][key] = dict(mean=float(np.mean(values)),
                                         std=float(np.std(values, ddof=1)) if len(values)>1 else None)
    save(args.output_dir / 'comparison.json', dict(scope='dev screening, not final test',
                                                 groups=results, aggregate=aggregate))
    for method, stats in aggregate.items():
        print(f'METHOD_SUMMARY method={method} seeds={stats["seeds"]} '
              f'clean_mean={stats["clean_miou"]["mean"]:.6f} '
              f'clean_std={stats["clean_miou"]["std"]} '
              f'corrupt_mean={stats["corruption_mean_miou"]["mean"]:.6f} '
              f'corrupt_std={stats["corruption_mean_miou"]["std"]}', flush=True)


def execute(args, started):
    noise.update(args, phase='preflight')
    protocol = preflight(args)
    results = []
    for name, path in reference_paths(args).items():
        row, _ = checked_reference(path)
        row.update(method=name.split('_seed')[0], seed=int(name.split('_seed')[1]))
        results.append(row)
    completed = []
    def run(method, seed):
        name = f'{method}_seed{seed}'
        args.window_progress = dict(groups_completed=len(completed), groups_possible=7,
                                    group_index=len(completed)+1, window='repair_v1')
        source = noise.load_source(args, seed)
        if method == 'R0':
            result = train_job(args, method, seed, source, protocol, None)
        else:
            refpath = (args.reference_dir / 'mixed' if seed == 42 else
                       args.noise_dir / 'R0_seed777' if seed == 777 else
                       args.output_dir / 'R0_seed2025')
            _, refs = checked_reference(refpath)
            result = noise.train_candidate(args, method, seed, source, protocol, refs)
        decoder_dim = source['args']['decoder_dim']
        del source
        completed.append(name)
        results.append(result)
        summarize(args, results)
        if method in ('S1', 'S2'):
            checkpoint = torch.load(args.output_dir / name / 'last.pth', map_location='cpu', weights_only=False)
            model = NoiseCMX(method, decoder_dim=decoder_dim).to(args.device)
            model.load_state_dict(checkpoint['model'], strict=True)
            save(args.output_dir / name / 'frontend_audit.json', frontend_audit(model, args))
            del model, checkpoint
        print(f'REPAIR_PROGRESS completed={len(completed)}/7_possible latest={name}', flush=True)
        return result
    run('D1', 777)
    candidates = [run(method, 42) for method in ('S1', 'S2')]
    private_diagnostic(args)
    decision_path = args.output_dir / 'decision.json'
    if decision_path.exists():
        decision = read(decision_path)
    else:
        selected = choose(candidates)
        remaining = args.budget_hours*60 - (time.time()-started)/60
        decision = dict(candidate=selected['method'] if selected else None,
                        remaining_minutes=remaining,
                        action='confirm' if selected and remaining >= 270 else 'screen_only')
        save(decision_path, decision)
    print(f'REPAIR_GATE {json.dumps(decision)}', flush=True)
    skipped = []
    if decision['action'] == 'confirm':
        winner = decision['candidate']
        queue = [(winner, 777), ('R0', 2025), ('D1', 2025), (winner, 2025)]
        for index, (method, seed) in enumerate(queue):
            exists = (args.output_dir / f'{method}_seed{seed}/result.json').exists()
            if not exists and args.budget_hours*60 - (time.time()-started)/60 < 65:
                skipped = [f'{m}_seed{s}' for m, s in queue[index:]]
                break
            run(method, seed)
    save(args.output_dir / 'complete.json', dict(training_runs=len(completed),
         evaluations=22*len(completed), private_evaluations=4, skipped=skipped,
         elapsed_minutes=(time.time()-started)/60, decision=decision))
    args.window_progress = {}
    noise.update(args, phase='complete', groups_completed=len(completed),
                 groups_planned=7 if decision['action']=='confirm' else 3, skipped=skipped)
    (args.output_dir / 'failure.json').unlink(missing_ok=True)
    print(f'REPAIR_COMPLETE new_training_groups={len(completed)} skipped={skipped}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = dict(phase1_dir='data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2',
                    nyuv2_dir='data/public_semseg/nyuv2/processed',
                    reference_dir='data/paper_benchmark_runs/mixed_control_v1',
                    noise_dir='data/paper_benchmark_runs/noise_window_v1',
                    output_dir='data/paper_benchmark_runs/repair_window_v1',
                    private_dir='data/new_data/merged',
                    private_checkpoint='data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B2/seed_42/best.pth')
    for name, value in defaults.items():
        parser.add_argument('--'+name.replace('_', '-'), type=Path, default=Path(value))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--budget-hours', type=float, default=8.)
    args = parser.parse_args()
    if not np.isfinite(args.budget_hours) or args.budget_hours < 5:
        parser.error('budget-hours must be finite and >=5')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / 'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname()), handle)
    started = time.time()
    try:
        execute(args, started)
    except BaseException:
        error = traceback.format_exc()
        save(args.output_dir / 'failure.json', dict(error=error))
        args.window_progress = {}
        noise.update(args, phase='failed', error=error)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
