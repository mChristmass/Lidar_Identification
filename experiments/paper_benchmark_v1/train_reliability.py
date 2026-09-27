"""R1/R2/R3 matched mixed-training pilot. NYUv2 train/dev only."""
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

from .corruptions_v2 import FAMILIES, RATES
from .diagnose_phase1 import DiagnosticData, digest, regional
from .reliability_model import ReliabilityCMX
from .run import seed_all
from .train_mixed_control import TrainingData, atomic_checkpoint

CONDITIONS = [('clean', 'medium')] + [(f, s) for f in FAMILIES for s in RATES]
EXPERIMENTS = ('R1', 'R2', 'R3')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def progress(args, **values):
    save(args.output_dir / 'status.json', {'updated_unix': time.time(), **values})


def validate_reference(args):
    ref = read(args.reference_dir / 'protocol.json')
    expected = dict(epochs=20, seed=42, batch_size=2, grad_accum=4, lr=1e-5,
                    checkpoint_sha256=digest(args.checkpoint),
                    train_manifest_sha256=digest(args.nyuv2_dir / 'manifests/train.jsonl'),
                    dev_manifest_sha256=digest(args.nyuv2_dir / 'manifests/dev.jsonl'))
    for key, value in expected.items():
        if ref.get(key) != value:
            raise ValueError(f'R0 reference mismatch: {key}')
    if read(args.reference_dir / 'complete.json') != dict(training_runs=2, epochs_each=20, evaluations=44):
        raise ValueError('R0 control suite is incomplete')
    references = {f'{f}_{s}': read(args.reference_dir / 'mixed/curves' / f'{f}_{s}.json')
                  for f, s in CONDITIONS}
    for row in references.values():
        if row.get('regime') != 'mixed' or row.get('epoch') != 20 or row.get('initialization_sha256') != expected['checkpoint_sha256']:
            raise ValueError('R0 curve provenance mismatch')
    # Freeze executable sources and reference curves; do not silently resume new code.
    sources = list(Path(__file__).parent.glob('*.py'))
    sources += list((Path(__file__).parent.parent / 'cmx_initial_transfer/official').glob('*.py'))
    protocol = dict(version=1, **expected, new_module_lr=args.module_lr,
                    selection='fixed epoch 20; no dev checkpoint selection',
                    augmentation=ref['augmentation'],
                    source_sha256={p.relative_to(Path(__file__).parent.parent).as_posix(): digest(p)
                                   for p in sorted(sources)},
                    reference_sha256={f'{f}_{s}': digest(args.reference_dir / 'mixed/curves' / f'{f}_{s}.json')
                                      for f, s in CONDITIONS})
    path = args.output_dir / 'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Protocol/code changed; use a new output directory')
    save(path, protocol)
    return protocol, references


def train_group(args, name, source, protocol, references):
    destination = args.output_dir / name
    destination.mkdir(exist_ok=True)
    last = destination / 'last.pth'
    result_path = destination / 'result.json'
    if result_path.exists():
        result = read(result_path)
        if not last.exists() or result['checkpoint_sha256'] != digest(last):
            raise ValueError(f'{name}: completed checkpoint changed/missing')
        print(f'GROUP_REUSE experiment={name}', flush=True)
        return result
    seed_all(42)
    model = ReliabilityCMX(name, decoder_dim=source['args']['decoder_dim']).to(args.device)
    model.load_baseline(source['model'])
    base, added = [], []
    for key, value in model.named_parameters():
        (base if key.startswith(('backbone.', 'decode_head.')) else added).append(value)
    optimizer = torch.optim.AdamW([{'params': base, 'lr': 1e-5, 'initial_lr': 1e-5},
                                  {'params': added, 'lr': args.module_lr, 'initial_lr': args.module_lr}],
                                 weight_decay=.01)
    amp = args.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start, history = 1, []
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state['protocol'] != protocol or state['experiment'] != name:
            raise ValueError('Resume checkpoint protocol mismatch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start, history = state['epoch'] + 1, state['history']
        del state
    dataset = TrainingData(args.nyuv2_dir, 'mixed', 42)
    for epoch in range(start, 21):
        seed_all(42 + epoch)
        dataset.epoch = epoch
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=False,
                            generator=torch.Generator().manual_seed(42 + epoch))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for index, (x, y) in enumerate(loader):
            factor = (1 - ((epoch - 1) + index / len(loader)) / 20) ** .9
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * factor
            window_samples = min(8, len(dataset) - (index // 4) * 8)
            x, y = x.to(args.device), y.to(args.device)
            with torch.autocast(device_type='cuda', enabled=amp):
                ce = torch.nn.functional.cross_entropy(model(x)['logits'], y, ignore_index=255)
                loss = ce * x.shape[0] / window_samples
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{name} epoch={epoch} batch={index}')
            scaler.scale(loss).backward()
            if (index + 1) % 4 == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                model.project_parameters()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(ce.detach()) * x.shape[0]
            if index % 25 == 0 or index + 1 == len(loader):
                progress(args, phase='training', experiment=name, group_index=EXPERIMENTS.index(name)+1,
                         groups_total=3, epoch=epoch, epochs=20, batch=index+1, batches=len(loader),
                         loss=loss_sum / min((index+1)*2, len(dataset)))
        controls = {key: value.detach().cpu().tolist() for key, value in model.named_parameters()
                    if key in ('depth_strength', 'intensity_strength', 'validity_projection.weight')}
        history.append(dict(epoch=epoch, loss=loss_sum/len(dataset), controls=controls))
        atomic_checkpoint(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scaler=scaler.state_dict(), epoch=epoch, history=history,
                          protocol=protocol, experiment=name))
        save(destination / 'history.json', {'epochs': history})
        print(f'PILOT_TRAIN group={EXPERIMENTS.index(name)+1}/3 experiment={name} '
              f'epoch={epoch}/20 loss={history[-1]["loss"]:.6f}', flush=True)
    checkpoint_hash = digest(last)
    model.eval()
    scores = []
    for index, (family, severity) in enumerate(CONDITIONS, 1):
        condition = f'{family}_{severity}'
        progress(args, phase='evaluation', experiment=name, condition=condition,
                 evaluation=index, evaluations=22, groups_total=3,
                 group_index=EXPERIMENTS.index(name)+1)
        path = destination / 'curves' / f'{condition}.json'
        result = read(path) if path.exists() else None
        if result is None or result.get('checkpoint_sha256') != checkpoint_hash:
            result = regional(model, DiagnosticData(args.nyuv2_dir, family, severity), args.device)
            result.update(checkpoint_sha256=checkpoint_hash, experiment=name, epoch=20,
                          condition=condition, initialization_sha256=protocol['checkpoint_sha256'])
        if result['degradation_audit'] != references[condition]['degradation_audit']:
            raise ValueError(f'Evaluation samples/budgets differ from R0: {condition}')
        save(path, result)
        score = result['regions']['all']['miou']
        baseline = references[condition]['regions']['all']['miou']
        scores.append(dict(condition=condition, miou=score, r0_miou=baseline,
                           gain_pp=100*(score-baseline)))
        print(f'PILOT_EVAL experiment={name} condition={index}/22 name={condition} '
              f'miou={score:.6f} gain_vs_R0_pp={100*(score-baseline):+.3f}', flush=True)
    result = dict(experiment=name, checkpoint_sha256=checkpoint_hash, scores=scores,
                  clean_miou=scores[0]['miou'], corruption_mean_miou=float(np.mean([r['miou'] for r in scores[1:]])),
                  corruption_mean_gain_pp=float(np.mean([r['gain_pp'] for r in scores[1:]])),
                  parameters=sum(p.numel() for p in model.parameters()),
                  added_parameters=sum(p.numel() for p in added), controls=history[-1]['controls'])
    save(result_path, result)
    print(f'GROUP_COMPLETE experiment={name} clean_miou={result["clean_miou"]:.6f} '
          f'corruption_mean={result["corruption_mean_miou"]:.6f} '
          f'gain_vs_R0_pp={result["corruption_mean_gain_pp"]:+.3f}', flush=True)
    del model, optimizer, scaler
    if amp:
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2/seed_42/best.pth'))
    parser.add_argument('--nyuv2-dir', type=Path, default=Path('data/public_semseg/nyuv2/processed'))
    parser.add_argument('--reference-dir', type=Path, default=Path('data/paper_benchmark_runs/mixed_control_v1'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/paper_benchmark_runs/reliability_pilot_v1'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--module-lr', type=float, default=1e-3)
    args = parser.parse_args()
    if not np.isfinite(args.module_lr) or args.module_lr <= 0:
        parser.error('--module-lr must be finite and positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / 'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname()), handle)
    try:
        progress(args, phase='preflight')
        protocol, references = validate_reference(args)
        source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        config = source['args']
        if any(config[k] != v for k, v in dict(dataset='nyuv2', model='CMX_B2', stage='tune',
                                              corruption='clean', seed=42, height=480, width=640).items()):
            raise ValueError('Expected original NYUv2 clean B2 seed42 tuning checkpoint')
        results = []
        for experiment in EXPERIMENTS:
            results.append(train_group(args, experiment, source, protocol, references))
            save(args.output_dir / 'comparison.json', {'scope': 'single-seed dev pilot', 'groups': results})
        save(args.output_dir / 'complete.json', dict(training_runs=3, epochs_each=20, evaluations=66))
        progress(args, phase='complete', groups_completed=3, evaluations_completed=66)
        (args.output_dir / 'failure.json').unlink(missing_ok=True)
        print('RELIABILITY_PILOT_COMPLETE groups=3/3 evaluations=66/66', flush=True)
    except BaseException:
        error = traceback.format_exc()
        save(args.output_dir / 'failure.json', {'error': error, 'time': time.time()})
        progress(args, phase='failed', error=error)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
