"""Five-hour paired intensity and reliability screen on frozen NYUv2 train/dev."""
import argparse
import json
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .corruptions_v2 import FAMILIES, RATES
from .diagnose_phase1 import DiagnosticData, digest, regional
from .noise_model import NoiseCMX
from .paired_intensity import PairedIntensityData
from .run import seed_all
from .run_fusion_window import train_job as train_r0
from .train_mixed_control import atomic_checkpoint
from .train_reliability import read, save

CONDITIONS = [('clean', 'medium')] + [(f, s) for f in FAMILIES for s in RATES]
METHODS = ('C1', 'D1', 'D2')
EPOCHS = 20


def update(args, **values):
    if hasattr(args, 'window_progress'):
        values.update(args.window_progress)
    save(args.output_dir / 'status.json', {'updated_unix': time.time(), **values})


def source_path(args, seed):
    return args.phase1_dir / f'seed_{seed}' / 'best.pth'


def condition_name(family, severity):
    return f'{family}_{severity}'


def references(path):
    return {condition_name(f, s): read(path / f'{condition_name(f, s)}.json')
            for f, s in CONDITIONS}


def preflight(args):
    old = read(args.reference_dir / 'protocol.json')
    if read(args.reference_dir / 'complete.json') != dict(training_runs=2, epochs_each=20, evaluations=44):
        raise ValueError('Seed42 mixed R0 is incomplete')
    manifest = args.nyuv2_dir / 'manifests'
    ck42, ck777 = source_path(args, 42), source_path(args, 777)
    if (old['checkpoint_sha256'] != digest(ck42)
            or old['train_manifest_sha256'] != digest(manifest / 'train.jsonl')
            or old['dev_manifest_sha256'] != digest(manifest / 'dev.jsonl')):
        raise ValueError('R0 source checkpoint or NYUv2 manifest differs')
    refs = references(args.reference_dir / 'mixed/curves')
    for row in refs.values():
        if row.get('regime') != 'mixed' or row.get('epoch') != 20:
            raise ValueError('R0 curve provenance mismatch')
    package = Path(__file__).parent
    tracked = [package / n for n in
               ('run_noise_window.py', 'noise_model.py', 'paired_intensity.py',
                'run_fusion_window.py', 'train_mixed_control.py', 'train_reliability.py',
                'corruptions.py', 'corruptions_v2.py', 'data.py', 'diagnose_phase1.py',
                'model.py', 'run.py')]
    tracked += sorted((package.parent / 'cmx_initial_transfer/official').glob('*.py'))
    protocol = dict(version=1, budget_hours=args.budget_hours,
                    train_seed42_then_confirmation_seed777=True,
                    epochs=EPOCHS, batch_size=2, grad_accum=4, base_lr=1e-5,
                    new_lr=1e-3, ce_weight=1., consistency_weight=.2,
                    denoise_weight=5., reliability_weight=.05,
                    temperature=2., reliability_target_scale=.12,
                    selection='fixed epoch20; no dev checkpoint reselection',
                    confirmation_rule='mean gain >=0.5pp OR photon/joint heavy gain >=1pp; clean drop <=0.5pp; >=150min remaining',
                    checkpoint_sha256={'42': digest(ck42), '777': digest(ck777)},
                    manifest_sha256={'train': digest(manifest / 'train.jsonl'),
                                     'dev': digest(manifest / 'dev.jsonl')},
                    reference_sha256={name: digest(args.reference_dir / 'mixed/curves' / f'{name}.json')
                                      for name in refs},
                    code_sha256={p.name if p.parent == package else
                                 'official/' + p.name: digest(p) for p in tracked})
    path = args.output_dir / 'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Output protocol changed; choose a new output directory')
    save(path, protocol)
    return protocol, refs


def load_source(args, seed):
    source = torch.load(source_path(args, seed), map_location='cpu', weights_only=False)
    expected = dict(dataset='nyuv2', model='CMX_B2', stage='tune',
                    corruption='clean', seed=seed, height=480, width=640)
    if any(source['args'].get(k) != v for k, v in expected.items()):
        raise ValueError(f'Incorrect clean B2 checkpoint for seed {seed}')
    return source


def consistency(student, teacher, labels, active):
    mask = (labels != 255) & active[:, None, None]
    if not bool(mask.any()):
        return student.sum() * 0
    p = (teacher.detach().float() / 2).softmax(1)
    logq = (student.float() / 2).log_softmax(1)
    divergence = F.kl_div(logq, p, reduction='none').sum(1) * 4
    return divergence[mask].mean()


def weighted_intensity_loss(per_sample, active):
    # Roughly 1/7 of mixed samples contain intensity noise. Keep a small
    # identity constraint on the others without swamping noisy supervision.
    weight = torch.where(active, 1., .05).to(per_sample.dtype)
    return (per_sample * weight).mean()


def selective_loss(prediction, clean_view, active, method):
    applied = (prediction['restored_intensity'].float() -
               clean_view[:, :1].float()).square().mean((1, 2, 3))
    raw = (prediction['raw_restored_intensity'].float() -
           clean_view[:, :1].float()).square().mean((1, 2, 3))
    # Inactive input intensity is exactly the clean target. Increase its
    # identity weight from D1's .05 to 1; keep noisy reconstruction weight 1.
    reconstruction = torch.where(active, .5*(raw+applied), applied).mean()
    gate_loss = reconstruction.new_zeros(())
    if method == 'S2':
        per_sample = F.binary_cross_entropy_with_logits(
            prediction['noise_gate_logits'].float().flatten(),
            active.float(), reduction='none')
        gate_loss = (per_sample * torch.where(active, 1., .2)).mean()
    return reconstruction, gate_loss


@torch.no_grad()
def confidence_audit(model, root, device):
    """Check whether D2 responds to dev stress; never used for selection."""
    output = {}
    clean = DiagnosticData(root, 'clean', 'medium')
    for family, severity in (('clean', 'medium'),
                             ('photon_proxy', 'heavy'), ('joint', 'heavy')):
        noisy = DiagnosticData(root, family, severity)
        accum = np.zeros(7, np.float64)
        for index in range(len(noisy)):
            item = noisy[index]
            reference = clean[index]
            if item['sample_id'] != reference['sample_id']:
                raise ValueError('Confidence-audit sample order differs')
            prediction = model(item['inputs'][None].to(device))['intensity_confidence']
            q = prediction.float().cpu().numpy().ravel().astype(np.float64)
            target = np.exp(-np.abs(
                item['inputs'][0].numpy().ravel().astype(np.float64) -
                reference['inputs'][0].numpy().ravel().astype(np.float64)) / .12)
            accum += [len(q), q.sum(), target.sum(), (q*q).sum(),
                      (target*target).sum(), (q*target).sum(),
                      (q < .5).sum()]
        n, qsum, tsum, q2, t2, qt, below = accum
        covariance = qt/n - (qsum/n)*(tsum/n)
        variance = (q2/n - (qsum/n)**2)*(t2/n - (tsum/n)**2)
        output[condition_name(family, severity)] = dict(
            predicted_mean=float(qsum/n), proxy_target_mean=float(tsum/n),
            predicted_fraction_below_half=float(below/n),
            proxy_correlation=float(covariance/np.sqrt(variance))
            if variance > 1e-12 else None)
    return output


def train_candidate(args, method, seed, source, protocol, refs):
    group_name = f'{method}_seed{seed}'
    destination = args.output_dir / group_name
    destination.mkdir(exist_ok=True)
    last = destination / 'last.pth'
    result_path = destination / 'result.json'
    if result_path.exists():
        result = read(result_path)
        if (not last.exists() or result['checkpoint_sha256'] != digest(last)
                or len(result['scores']) != 22):
            raise ValueError(f'Completed artifact changed: {group_name}')
        print(f'GROUP_REUSE {group_name}', flush=True)
        return result
    seed_all(seed)
    model = NoiseCMX(method, decoder_dim=source['args']['decoder_dim']).to(args.device)
    model.load_baseline(source['model'])
    base, added = [], []
    for name, parameter in model.named_parameters():
        (base if name.startswith(('backbone.', 'decode_head.')) else added).append(parameter)
    groups = [{'params': base, 'lr': 1e-5, 'initial_lr': 1e-5}]
    if added:
        groups.append({'params': added, 'lr': 1e-3, 'initial_lr': 1e-3})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
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
    dataset = PairedIntensityData(args.nyuv2_dir, seed)
    group_index = {'C1': 1, 'D1': 2, 'D2': 3, 'S1': 2, 'S2': 3}[method] if seed == 42 else 5
    for epoch in range(start, EPOCHS + 1):
        seed_all(seed + epoch)
        dataset.epoch = epoch
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
                            pin_memory=False,
                            generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = dict(ce=0., consistency=0., denoise=0., reliability=0.)
        for index, (x, y, clean_view, active) in enumerate(loader):
            factor = (1 - ((epoch - 1) + index/len(loader)) / EPOCHS) ** .9
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * factor
            window_samples = min(8, len(dataset) - (index // 4)*8)
            x, y, clean_view = (value.to(args.device) for value in (x, y, clean_view))
            active = active.to(args.device)
            with torch.autocast(device_type='cuda', enabled=amp):
                prediction = model(x)
                ce = F.cross_entropy(prediction['logits'], y, ignore_index=255)
            kl = ce.new_zeros(())
            if bool(active.any()):
                model.eval()
                with torch.no_grad(), torch.autocast(device_type='cuda', enabled=amp):
                    teacher = model(clean_view)['logits']
                model.train()
                kl = consistency(prediction['logits'], teacher, y, active)
            mse = ce.new_zeros(())
            confidence_loss = ce.new_zeros(())
            if method != 'C1':
                squared = (prediction['restored_intensity'].float() -
                           clean_view[:, :1].float()).square().mean(dim=(1, 2, 3))
                mse = weighted_intensity_loss(squared, active)
            if method == 'D2':
                target = torch.exp(-(x[:, :1].float() -
                                     clean_view[:, :1].float()).abs() / .12)
                per_pixel = F.binary_cross_entropy(
                    prediction['intensity_confidence'].float(), target, reduction='none')
                confidence_loss = weighted_intensity_loss(
                    per_pixel.mean(dim=(1, 2, 3)), active)
            if method in ('S1', 'S2'):
                mse, confidence_loss = selective_loss(prediction, clean_view, active, method)
            combined = ce + .2*kl + 5*mse + (.1 if method == 'S2' else .05)*confidence_loss
            loss = combined * x.shape[0] / window_samples
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{group_name} epoch={epoch} batch={index}')
            scaler.scale(loss).backward()
            if (index + 1) % 4 == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                model.project_parameters()
                optimizer.zero_grad(set_to_none=True)
            for key, value in (('ce', ce), ('consistency', kl), ('denoise', mse),
                               ('reliability', confidence_loss)):
                totals[key] += float(value.detach()) * x.shape[0]
            if index % 25 == 0 or index + 1 == len(loader):
                update(args, phase='training', group=group_name,
                       group_index=group_index, groups_possible=5,
                       epoch=epoch, epochs=EPOCHS, batch=index+1, batches=len(loader),
                       ce=totals['ce']/min((index+1)*2, len(dataset)))
        controls = ({'fusion_strength': model.fusion_strength.detach().cpu().tolist()}
                    if method == 'D2' else {})
        history.append(dict(epoch=epoch, losses={k: v/len(dataset)
                                                   for k, v in totals.items()},
                            controls=controls))
        atomic_checkpoint(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scaler=scaler.state_dict(), epoch=epoch, history=history,
                          protocol=protocol, group=group_name))
        save(destination / 'history.json', {'epochs': history})
        print(f'NOISE_TRAIN group={group_name} epoch={epoch}/{EPOCHS} '
              f'ce={history[-1]["losses"]["ce"]:.6f}', flush=True)
    fingerprint = digest(last)
    model.eval()
    scores = []
    for index, (family, severity) in enumerate(CONDITIONS, 1):
        condition = condition_name(family, severity)
        update(args, phase='evaluation', group=group_name,
               group_index=group_index, groups_possible=5,
               condition=condition, evaluation=index, evaluations=22)
        path = destination / 'curves' / f'{condition}.json'
        result = read(path) if path.exists() else None
        if result is None or result.get('checkpoint_sha256') != fingerprint:
            result = regional(model, DiagnosticData(args.nyuv2_dir, family, severity),
                              args.device)
            result.update(checkpoint_sha256=fingerprint, group=group_name,
                          condition=condition, epoch=EPOCHS)
        if result['degradation_audit'] != refs[condition]['degradation_audit']:
            raise ValueError(f'Changed sample/budget: {group_name}/{condition}')
        save(path, result)
        score, base_score = (result['regions']['all']['miou'],
                             refs[condition]['regions']['all']['miou'])
        scores.append(dict(condition=condition, miou=score, reference_miou=base_score,
                           gain_pp=100*(score-base_score)))
        print(f'NOISE_EVAL group={group_name} condition={index}/22 {condition} '
              f'miou={score:.6f} gain_pp={scores[-1]["gain_pp"]:+.3f}', flush=True)
    heavy = {item['condition']: item['gain_pp'] for item in scores}
    result = dict(group=group_name, method=method, seed=seed,
                  checkpoint_sha256=fingerprint, scores=scores,
                  clean_miou=scores[0]['miou'],
                  corruption_mean_miou=float(np.mean([r['miou'] for r in scores[1:]])),
                  corruption_mean_gain_pp=float(np.mean([r['gain_pp'] for r in scores[1:]])),
                  photon_heavy_gain_pp=heavy['photon_proxy_heavy'],
                  joint_heavy_gain_pp=heavy['joint_heavy'],
                  added_parameters=sum(p.numel() for p in added), controls=history[-1]['controls'])
    if method == 'D2':
        update(args, phase='confidence_audit', group=group_name)
        result['confidence_audit'] = confidence_audit(model, args.nyuv2_dir, args.device)
    save(result_path, result)
    print(f'GROUP_COMPLETE {group_name} clean={result["clean_miou"]:.6f} '
          f'corrupt_mean={result["corruption_mean_miou"]:.6f} '
          f'gain_pp={result["corruption_mean_gain_pp"]:+.3f}', flush=True)
    del model, optimizer, scaler
    if amp:
        torch.cuda.empty_cache()
    return result


def choose(results, refs):
    clean_ref = refs['clean_medium']['regions']['all']['miou']
    eligible = [r for r in results if
                100*(r['clean_miou']-clean_ref) >= -.5
                and (r['corruption_mean_gain_pp'] >= .5
                     or r['photon_heavy_gain_pp'] >= 1.
                     or r['joint_heavy_gain_pp'] >= 1.)]
    return max(eligible, key=lambda r: r['corruption_mean_gain_pp']) if eligible else None


def execute(args, started):
    update(args, phase='preflight')
    protocol, refs = preflight(args)
    source42 = load_source(args, 42)
    results = []
    for method in METHODS:
        results.append(train_candidate(args, method, 42, source42, protocol, refs))
        save(args.output_dir / 'comparison.json', {'scope': 'single-seed NYUv2 dev screen',
                                                   'groups': results})
    selected = choose(results, refs)
    remaining_minutes = (args.budget_hours*3600 - (time.time()-started))/60
    action = ('run_second_seed' if selected and remaining_minutes >= 150
              else 'stop_after_screen')
    decision = dict(action=action, candidate=selected['method'] if selected else None,
                    remaining_minutes=remaining_minutes, rule=protocol['confirmation_rule'])
    save(args.output_dir / 'decision.json', decision)
    print(f'CONFIRMATION_GATE action={action} candidate={decision["candidate"]} '
          f'remaining_minutes={remaining_minutes:.1f}', flush=True)
    if action == 'run_second_seed':
        del source42
        source777 = load_source(args, 777)
        r0 = train_r0(args, 'R0', 777, source777, protocol, None)
        results.append(r0)
        save(args.output_dir / 'comparison.json', {'scope': 'NYUv2 dev screen and paired second seed',
                                                   'groups': results})
        ref777 = references(args.output_dir / 'R0_seed777/curves')
        results.append(train_candidate(args, selected['method'], 777,
                                       source777, protocol, ref777))
        save(args.output_dir / 'comparison.json', {'scope': 'NYUv2 dev screen and paired second seed',
                                                   'groups': results})
    save(args.output_dir / 'complete.json',
         dict(training_runs=len(results), evaluations=22*len(results),
              elapsed_minutes=(time.time()-started)/60, gate=decision))
    update(args, phase='complete', groups_completed=len(results), gate_action=action)
    (args.output_dir / 'failure.json').unlink(missing_ok=True)
    print(f'NOISE_WINDOW_COMPLETE groups={len(results)} evaluations={22*len(results)}',
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase1-dir', type=Path,
                        default=Path('data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2'))
    parser.add_argument('--nyuv2-dir', type=Path,
                        default=Path('data/public_semseg/nyuv2/processed'))
    parser.add_argument('--reference-dir', type=Path,
                        default=Path('data/paper_benchmark_runs/mixed_control_v1'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/paper_benchmark_runs/noise_window_v1'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--budget-hours', type=float, default=5.)
    args = parser.parse_args()
    if not np.isfinite(args.budget_hours) or not 3 <= args.budget_hours <= 5:
        parser.error('--budget-hours must be in [3,5]')
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
