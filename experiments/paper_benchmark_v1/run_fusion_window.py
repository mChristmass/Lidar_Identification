"""One-launch 3-5 hour fusion pilot with a predeclared confirmation gate."""
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
from .diagnose_modalities import run_diagnostic
from .diagnose_phase1 import DiagnosticData, digest, regional
from .fusion_pilot_model import FusionPilotCMX
from .model import CMXPaper
from .run import seed_all
from .train_mixed_control import TrainingData, atomic_checkpoint
from .train_reliability import read, save

CONDITIONS = [('clean', 'medium')] + [(f, s) for f in FAMILIES for s in RATES]
BASE_LR = 1e-5
NEW_LR = 1e-3
EPOCHS = 20


def update(args, **values):
    if hasattr(args, 'window_progress'):
        values.update(args.window_progress)
    save(args.output_dir / 'status.json', {'updated_unix': time.time(), **values})


def checkpoint_path(args, seed):
    return args.phase1_dir / f'seed_{seed}' / 'best.pth'


def condition_key(family, severity):
    return f'{family}_{severity}'


def reference_curves(path):
    return {condition_key(f, s): read(path / f'{condition_key(f, s)}.json')
            for f, s in CONDITIONS}


def preflight(args):
    if not (3 <= args.budget_hours <= 5):
        raise ValueError('This launcher is fixed to a 3-5 hour server window')
    required = [Path(__file__), Path(__file__).parent / 'fusion_pilot_model.py',
                Path(__file__).parent / 'diagnose_modalities.py',
                Path(__file__).parent / 'train_mixed_control.py',
                Path(__file__).parent / 'train_reliability.py',
                Path(__file__).parent / 'diagnose_phase1.py',
                Path(__file__).parent / 'data.py',
                Path(__file__).parent / 'corruptions.py',
                Path(__file__).parent / 'corruptions_v2.py',
                Path(__file__).parent / 'model.py',
                Path(__file__).parent / 'run.py']
    required += sorted((Path(__file__).parent.parent / 'cmx_initial_transfer/official').glob('*.py'))
    manifest = args.nyuv2_dir / 'manifests'
    ck42, ck777 = checkpoint_path(args, 42), checkpoint_path(args, 777)
    mixed_last = args.reference_dir / 'mixed/last.pth'
    if read(args.reference_dir / 'complete.json') != dict(training_runs=2, epochs_each=20, evaluations=44):
        raise ValueError('Mixed R0 reference is incomplete')
    ref_protocol = read(args.reference_dir / 'protocol.json')
    if (ref_protocol['checkpoint_sha256'] != digest(ck42)
            or ref_protocol['train_manifest_sha256'] != digest(manifest / 'train.jsonl')
            or ref_protocol['dev_manifest_sha256'] != digest(manifest / 'dev.jsonl')):
        raise ValueError('Frozen R0 source or NYUv2 manifest changed')
    baseline = reference_curves(args.reference_dir / 'mixed/curves')
    for row in baseline.values():
        if row.get('regime') != 'mixed' or row.get('epoch') != 20:
            raise ValueError('Wrong R0 reference curves')
    state = torch.load(mixed_last, map_location='cpu', weights_only=False)
    if state['epoch'] != 20 or len(state['history']) != 20:
        raise ValueError('R0 mixed checkpoint is incomplete')
    del state
    checksums = {str(p.relative_to(Path(__file__).parent.parent)).replace('\\', '/'): digest(p)
                 for p in required}
    protocol = dict(version=1, budget_hours=args.budget_hours, epochs=EPOCHS,
                    base_lr=BASE_LR, new_lr=NEW_LR, batch_size=2, grad_accum=4,
                    weight_decay=.01, train_split='715 frozen tune frames',
                    selection='last epoch, no dev checkpoint selection',
                    confirmation_rule='seed42 gain >=0.5pp corruption mean OR >=1pp joint heavy, clean drop <=0.5pp; >=150min remaining',
                    checkpoint_sha256={'42': digest(ck42), '777': digest(ck777),
                                       'R0_mixed_42': digest(mixed_last)},
                    manifests_sha256={'train': digest(manifest / 'train.jsonl'),
                                      'dev': digest(manifest / 'dev.jsonl')},
                    reference_sha256={name: digest(args.reference_dir / 'mixed/curves' / f'{name}.json')
                                      for name in baseline},
                    code_sha256=checksums)
    path = args.output_dir / 'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Code, data, budget, or checkpoint changed; choose a new output directory')
    save(path, protocol)
    return protocol, baseline


def load_source(args, seed):
    source = torch.load(checkpoint_path(args, seed), map_location='cpu', weights_only=False)
    expected = dict(dataset='nyuv2', model='CMX_B2', stage='tune',
                    corruption='clean', seed=seed, height=480, width=640)
    if any(source['args'].get(k) != v for k, v in expected.items()):
        raise ValueError(f'Wrong clean source checkpoint for seed {seed}')
    return source


def train_job(args, name, seed, source, protocol, refs):
    group_name = f'{name}_seed{seed}'
    group_index = {('F1', 42): 1, ('F2', 42): 2,
                    ('R0', 777): 3, ('R0', 2025): 5,
                    ('F1', 777): 4, ('F2', 777): 4}[(name, seed)]
    destination = args.output_dir / group_name
    destination.mkdir(exist_ok=True)
    last = destination / 'last.pth'
    result_path = destination / 'result.json'
    if result_path.exists():
        result = read(result_path)
        if not last.exists() or result['checkpoint_sha256'] != digest(last):
            raise ValueError(f'Completed checkpoint changed: {group_name}')
        if len(result['scores']) != 22:
            raise ValueError(f'Incomplete saved evaluation: {group_name}')
        print(f'GROUP_REUSE {group_name}', flush=True)
        return result
    seed_all(seed)
    if name == 'R0':
        model = CMXPaper('CMX_B2', 40, 640, source['args']['decoder_dim']).to(args.device)
        model.load_state_dict(source['model'], strict=True)
        optimizer = torch.optim.AdamW([{'params': list(model.parameters()),
                                        'lr': BASE_LR, 'initial_lr': BASE_LR}], weight_decay=.01)
        added_count = 0
    else:
        model = FusionPilotCMX(name, decoder_dim=source['args']['decoder_dim']).to(args.device)
        model.load_baseline(source['model'])
        added = [p for k, p in model.named_parameters()
                 if not k.startswith(('backbone.', 'decode_head.'))]
        base = [p for k, p in model.named_parameters()
                if k.startswith(('backbone.', 'decode_head.'))]
        optimizer = torch.optim.AdamW([{'params': base, 'lr': BASE_LR, 'initial_lr': BASE_LR},
                                       {'params': added, 'lr': NEW_LR, 'initial_lr': NEW_LR}],
                                      weight_decay=.01)
        added_count = sum(p.numel() for p in added)
    amp = args.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, start = [], 1
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state['protocol'] != protocol or state['group'] != group_name:
            raise ValueError('Resume protocol/group mismatch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        history, start = state['history'], state['epoch'] + 1
        del state
    dataset = TrainingData(args.nyuv2_dir, 'mixed', seed)
    for epoch in range(start, EPOCHS + 1):
        seed_all(seed + epoch)
        dataset.epoch = epoch
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
                            pin_memory=False,
                            generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for index, (x, y) in enumerate(loader):
            factor = (1 - ((epoch - 1) + index / len(loader)) / EPOCHS) ** .9
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * factor
            window_samples = min(8, len(dataset) - (index // 4) * 8)
            x, y = x.to(args.device), y.to(args.device)
            with torch.autocast(device_type='cuda', enabled=amp):
                ce = torch.nn.functional.cross_entropy(model(x)['logits'], y, ignore_index=255)
                loss = ce * x.shape[0] / window_samples
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{group_name} epoch={epoch} batch={index}')
            scaler.scale(loss).backward()
            if (index + 1) % 4 == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                if name != 'R0':
                    model.project_parameters()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(ce.detach()) * x.shape[0]
            if index % 25 == 0 or index + 1 == len(loader):
                update(args, phase='training', group=group_name,
                       group_index=group_index, groups_possible=4, epoch=epoch,
                       epochs=EPOCHS, batch=index + 1, batches=len(loader),
                       loss=loss_sum/min((index + 1)*2, len(dataset)))
        controls = ({key: value.detach().cpu().tolist()
                     for key, value in model.named_parameters()
                     if key in ('cross_gate', 'propagation_gate')} if name != 'R0' else {})
        history.append(dict(epoch=epoch, loss=loss_sum/len(dataset), controls=controls))
        atomic_checkpoint(last, dict(model=model.state_dict(),
                          optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                          epoch=epoch, history=history, protocol=protocol, group=group_name))
        save(destination / 'history.json', {'epochs': history})
        print(f'FUSION_TRAIN group={group_name} epoch={epoch}/{EPOCHS} '
              f'loss={history[-1]["loss"]:.6f}', flush=True)
    fingerprint = digest(last)
    model.eval()
    scores = []
    for index, (family, severity) in enumerate(CONDITIONS, 1):
        condition = condition_key(family, severity)
        update(args, phase='evaluation', group=group_name,
               group_index=group_index, groups_possible=4, condition=condition,
               evaluation=index, evaluations=22)
        path = destination / 'curves' / f'{condition}.json'
        result = read(path) if path.exists() else None
        if result is None or result.get('checkpoint_sha256') != fingerprint:
            result = regional(model, DiagnosticData(args.nyuv2_dir, family, severity), args.device)
            result.update(checkpoint_sha256=fingerprint, group=group_name,
                          condition=condition, epoch=EPOCHS)
        if refs is not None and result['degradation_audit'] != refs[condition]['degradation_audit']:
            raise ValueError(f'Condition sample/budget mismatch: {group_name}/{condition}')
        save(path, result)
        score = result['regions']['all']['miou']
        baseline = refs[condition]['regions']['all']['miou'] if refs else None
        scores.append(dict(condition=condition, miou=score,
                           reference_miou=baseline,
                           gain_pp=100*(score-baseline) if baseline is not None else None))
        print(f'FUSION_EVAL group={group_name} condition={index}/22 {condition} '
              f'miou={score:.6f} gain_pp={scores[-1]["gain_pp"]}', flush=True)
    result = dict(group=group_name, seed=seed, method=name, scores=scores,
                  checkpoint_sha256=fingerprint,
                  clean_miou=scores[0]['miou'],
                  corruption_mean_miou=float(np.mean([r['miou'] for r in scores[1:]])),
                  corruption_mean_gain_pp=(float(np.mean([r['gain_pp'] for r in scores[1:]]))
                                           if refs is not None else None),
                  joint_heavy_gain_pp=(next(r['gain_pp'] for r in scores
                                            if r['condition'] == 'joint_heavy')
                                       if refs is not None else None),
                  added_parameters=added_count,
                  controls=history[-1]['controls'])
    save(result_path, result)
    print(f'GROUP_COMPLETE {group_name} clean={result["clean_miou"]:.6f} '
          f'corrupt_mean={result["corruption_mean_miou"]:.6f} '
          f'gain_pp={result["corruption_mean_gain_pp"]}', flush=True)
    del model, optimizer, scaler
    if amp:
        torch.cuda.empty_cache()
    return result


def select_confirmation(results, reference):
    clean_ref = reference['clean_medium']['regions']['all']['miou']
    eligible = []
    for result in results:
        clean_drop_pp = 100*(result['clean_miou'] - clean_ref)
        if clean_drop_pp >= -.5 and (result['corruption_mean_gain_pp'] >= .5
                                     or result['joint_heavy_gain_pp'] >= 1.):
            eligible.append(result)
    return max(eligible, key=lambda item: item['corruption_mean_gain_pp']) if eligible else None


def execute(args, started):
    update(args, phase='preflight')
    protocol, reference = preflight(args)
    source42 = load_source(args, 42)
    run_diagnostic(args.reference_dir / 'mixed/last.pth',
                   source42['args']['decoder_dim'], args.nyuv2_dir,
                   args.output_dir / 'diagnostic_curves',
                   args.reference_dir / 'mixed/curves', args.device,
                   lambda **status: update(args, **status))
    results = []
    for name in ('F1', 'F2'):
        result = train_job(args, name, 42, source42, protocol, reference)
        results.append(result)
        save(args.output_dir / 'comparison.json',
             {'scope': 'frozen NYUv2 dev pilot', 'groups': results})
    choice = select_confirmation(results, reference)
    remaining_minutes = (args.budget_hours*3600 - (time.time() - started))/60
    decision = dict(candidate=choice['method'] if choice else None,
                    remaining_minutes=remaining_minutes,
                    rule=protocol['confirmation_rule'],
                    action=('run_second_seed' if choice and remaining_minutes >= 150
                            else 'stop_after_screen'))
    save(args.output_dir / 'decision.json', decision)
    print(f'CONFIRMATION_GATE candidate={decision["candidate"]} '
          f'remaining_minutes={remaining_minutes:.1f} action={decision["action"]}', flush=True)
    if decision['action'] == 'run_second_seed':
        del source42
        source777 = load_source(args, 777)
        r0 = train_job(args, 'R0', 777, source777, protocol, None)
        results.append(r0)
        save(args.output_dir / 'comparison.json',
             {'scope': 'frozen NYUv2 dev pilot', 'groups': results})
        refs777 = reference_curves(args.output_dir / 'R0_seed777/curves')
        confirmed = train_job(args, choice['method'], 777, source777, protocol, refs777)
        results.append(confirmed)
        save(args.output_dir / 'comparison.json',
             {'scope': 'frozen NYUv2 dev pilot', 'groups': results})
    save(args.output_dir / 'complete.json',
         dict(training_runs=len(results), evaluations=22*len(results),
              diagnostic_evaluations=9, gate=decision, elapsed_minutes=(time.time()-started)/60))
    update(args, phase='complete', groups_completed=len(results),
           gate_action=decision['action'])
    (args.output_dir / 'failure.json').unlink(missing_ok=True)
    print(f'FUSION_WINDOW_COMPLETE groups={len(results)} '
          f'evaluations={22*len(results)} diagnostic=9', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase1-dir', type=Path,
                        default=Path('data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2'))
    parser.add_argument('--nyuv2-dir', type=Path,
                        default=Path('data/public_semseg/nyuv2/processed'))
    parser.add_argument('--reference-dir', type=Path,
                        default=Path('data/paper_benchmark_runs/mixed_control_v1'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/paper_benchmark_runs/fusion_window_v1'))
    parser.add_argument('--budget-hours', type=float, default=4.5)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not np.isfinite(args.budget_hours) or not 3 <= args.budget_hours <= 5:
        parser.error('--budget-hours must be between 3 and 5')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / 'RUNNING.lock'
    with lock.open('x', encoding='utf-8') as handle:
        json.dump({'pid': os.getpid(), 'host': socket.gethostname()}, handle)
    started = time.time()
    try:
        execute(args, started)
    except BaseException:
        error = traceback.format_exc()
        save(args.output_dir / 'failure.json', {'error': error, 'time': time.time()})
        update(args, phase='failed', error=error)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
