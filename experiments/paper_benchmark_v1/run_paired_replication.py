"""Six-hour fixed20 PAIR/CONS replication at seeds777/2025, both domains."""
import argparse
import gc
import json
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PrivatePaperDataset, read_manifest
from .diagnose_phase1 import DiagnosticData, digest, regional
from .paired_model import (PairedCMX, backward_objective, VIEW_CONFIG,
                           CONS_WEIGHT, MASK_WEIGHT, RAMP_EPOCHS)
from .run_structure_window import (PrivateTrain, source_path, load_source, audit_inputs,
                                   component_counts, private_details)
from .run import seed_all, evaluate
from .run_noise_window import CONDITIONS
from .train_mixed_control import TrainingData, atomic_checkpoint
from .train_reliability import read, save

METHODS = ('PAIR', 'CONS')
SEEDS = (777, 2025)
EPOCHS = 20


def update(args, **state):
    if hasattr(args, 'session_started'):
        state['elapsed_minutes'] = round((time.time()-args.session_started)/60, 1)
        state['budget_remaining_minutes'] = round(max(0.,args.budget_hours*60-state['elapsed_minutes']),1)
    save(args.output_dir/'status.json', dict(updated_unix=time.time(),
         groups_completed=getattr(args, 'completed', 0), groups_possible=8, **state))


def preflight(args):
    split = args.private_dir/'paper_split_v1'
    train_ids, dev_ids = [np.load(split/f'{s}_indices.npy') for s in ('train', 'dev')]
    if len(train_ids)!=970 or len(dev_ids)!=137 or np.intersect1d(train_ids, dev_ids).size:
        raise ValueError('Private split differs from frozen970/137')
    prior = read(args.repair_dir/'protocol.json')
    files = {f'private/{k}': (args.private_dir/Path(k.replace('\\', '/')))
             for k in prior['private_sha256'] if k!='checkpoint'}
    # Include ALL used training labels as well as dev labels. No final-test labels.
    files.update({f'private/label/{int(i)}.png': args.private_dir/'label'/f'{int(i)}.png' for i in train_ids})
    for part in ('train', 'dev'):
        manifest = args.nyuv2_dir/'manifests'/f'{part}.jsonl'
        if digest(manifest)!=prior['manifests'][part]:
            raise ValueError('NYUv2 frozen manifest changed')
        files[f'nyuv2/{part}_manifest'] = manifest
        for row in read_manifest(manifest):
            for key in ('intensity', 'depth', 'label'):
                files['nyuv2/'+row[key]] = args.nyuv2_dir/row[key]
    for dataset in ('private', 'nyuv2'):
        for seed in (42, *SEEDS):
            files[f'source/{dataset}/{seed}'] = source_path(args, dataset, seed)
            source = load_source(args, dataset, seed)
            del source
    fingerprints = {key: digest(path) for key, path in files.items()}
    for key, value in prior['private_sha256'].items():
        if key!='checkpoint' and fingerprints['private/'+key]!=value:
            raise ValueError(f'Private frozen data changed: {key}')
    package = Path(__file__).parent
    code = sorted(package.glob('*.py'))+sorted((package.parent/'cmx_initial_transfer/official').glob('*.py'))
    reference_files, args.reference_results = verify_reference(args, fingerprints)
    protocol = dict(version='paired_replication_v1', fingerprints=fingerprints,
                    code={str(p.relative_to(package.parent)): digest(p) for p in code},
                    epochs=EPOCHS, methods=list(METHODS), seeds=list(SEEDS), budget_hours=args.budget_hours,
                    scope='dev-only broad screen; BOTH datasets participate; no efficacy gate',
                    module_design='Unchanged PAIR and CONS only; paired_model reused without modifications',
                    reference_files=reference_files, view_config=VIEW_CONFIG,
                    consistency_weight=CONS_WEIGHT, completion_weight=MASK_WEIGHT, ramp_epochs=RAMP_EPOCHS,
                    clean_reconstruction_target_used=False,
                    sources='private best B1 / public clean best B2, identical per dataset/seed across methods',
                    training='reference private natural flip/public mixed v2; paired same extra noise and block depth masking; fixed final epoch',
                    optimizer='AdamW base1e-5 added1e-3 wd.01 poly.9 clip1; batch8 private / batch2 accum4 public',
                    observed_inputs=['intensity','depth','validity'], interpretation='observed semantic/feature targets, not clean image or physical depth reconstruction',
                    schedule='fixed8 new groups, 20epochs; mandatory seeds777/2025; soft6h, no score gate or truncation')
    path = args.output_dir/'protocol.json'
    if path.exists() and read(path)!=protocol:
        raise ValueError('Protocol/input/code changed: use a new output directory')
    save(path, protocol)
    TrainingData(args.nyuv2_dir, 'mixed', 42)
    return protocol


@torch.no_grad()
def probe_modules(args, model, dataset, seed, destination):
    """Fixed train samples, on/off within the SAME learned base; not causal attribution."""
    regimes = ('clean', 'mixed') if dataset=='nyuv2' else ('natural',)
    rows = []
    model.eval()
    for regime in regimes:
        data = PrivateTrain(args.private_dir, seed) if dataset=='private' else TrainingData(args.nyuv2_dir, regime, seed)
        data.epoch = EPOCHS
        for index in np.linspace(0,len(data)-1,6,dtype=int):
            x, _ = data[int(index)]
            x = x[None].to(args.device)
            try:
                model.adapters_enabled = True
                model.audit_enabled = True
                on = model(x)['logits'].float()
                stages = model.audit_rows
                model.adapters_enabled = False
                off = model(x)['logits'].float()
                row=dict(regime=regime,train_index=int(index),stages=stages,
                    logit_mean_abs_change=float((on-off).abs().mean()),
                    prediction_changed_fraction=float((on.argmax(1)!=off.argmax(1)).float().mean()))
                rows.append(row)
            finally:
                model.adapters_enabled = True
                model.audit_enabled = False
                model.capture_features = False
    save(destination/'module_probe.json',dict(rows=rows,
         note='Six fixed training samples/regime; mixed may draw clean. No labels used. Not full-dev ablation.'))

def train_group(args, dataset, method, seed, protocol):
    name = f'{dataset}/{method}_seed{seed}'
    destination = args.output_dir/name
    destination.mkdir(parents=True, exist_ok=True)
    last, result_path = destination/'last.pth', destination/'result.json'
    if result_path.exists():
        result = read(result_path)
        if result['checkpoint_sha256']!=digest(last):
            raise ValueError(f'Completed checkpoint changed: {name}')
        return result
    started = time.time()
    source = load_source(args, dataset, seed)
    seed_all(seed)
    model = PairedCMX(method, source['args']['model'], 2 if dataset=='private' else 40,
                           128 if dataset=='private' else 640, source['args']['decoder_dim']).to(args.device)
    model.load_baseline(source['model'])
    del source
    base, added = [], []
    for key, value in model.named_parameters():
        (added if key.startswith('adapters.') else base).append(value)
    groups = [dict(params=base, lr=1e-5, initial_lr=1e-5)]
    if added:
        groups.append(dict(params=added, lr=1e-3, initial_lr=1e-3))
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    initial_added = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if k.startswith('adapters.')}
    prior_minutes = 0.
    amp = args.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, start = [], 1
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state['protocol']!=protocol or state['group']!=name:
            raise ValueError('Resume protocol/group mismatch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        history, start = state['history'], state['epoch']+1
        prior_minutes = state.get('elapsed_minutes', 0.)
        del state
    data = PrivateTrain(args.private_dir, seed) if dataset=='private' else TrainingData(args.nyuv2_dir, 'mixed', seed)
    batch, accum = (8,1) if dataset=='private' else (2,4)
    for epoch in range(start, EPOCHS+1):
        data.epoch = epoch
        seed_all(seed+epoch)
        loader = DataLoader(data, batch_size=batch, shuffle=True, num_workers=0, pin_memory=False,
                            generator=torch.Generator().manual_seed(seed+epoch))
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        totals = {}
        steps, skipped_steps, added_grad_max = 0, 0, 0.
        for index, (x,y) in enumerate(loader):
            x, y = x.to(args.device), y.to(args.device)
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr']*(1-((epoch-1)+index/len(loader))/EPOCHS)**.9
            window_samples = min(batch*accum, len(data)-(index//accum)*batch*accum)
            stats = backward_objective(model, x, y, dataset, method, epoch,
                                       seed*10000000+epoch*10000+index, scaler, len(x)/window_samples)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.) + value*len(x)
            if (index+1)%accum==0 or index+1==len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if added:
                    grad = max((p.grad.detach().float().abs().max().item() for p in added if p.grad is not None), default=0.)
                    if np.isfinite(grad):
                        added_grad_max = max(added_grad_max, grad)
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                skipped_steps += int(scaler.get_scale() < old_scale)
                steps += 1
                optimizer.zero_grad(set_to_none=True)
            loss_sum += stats['loss']*len(x)
            if index%25==0 or index+1==len(loader):
                update(args, phase='training', group=name, epoch=epoch, epochs=EPOCHS,
                       batch=index+1, batches=len(loader), loss=loss_sum/min((index+1)*batch,len(data)))
        history.append(dict(epoch=epoch, **{k:v/len(data) for k,v in totals.items()}, added_grad_max=added_grad_max,
                            attempted_steps=steps, skipped_amp_steps=skipped_steps))
        if steps == skipped_steps:
            raise RuntimeError('All optimizer steps skipped in this epoch')
        atomic_checkpoint(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scaler=scaler.state_dict(), history=history, epoch=epoch, protocol=protocol, group=name,
                          elapsed_minutes=prior_minutes+(time.time()-started)/60))
        save(destination/'history.json', dict(epochs=history))
        print(f'REPLICATION_EPOCH {name} epoch={epoch}/{EPOCHS} loss={history[-1]["loss"]:.6f}', flush=True)
    model.eval()
    parameter_audit = []
    for key, value in model.named_parameters():
        if key in initial_added:
            delta = value.detach().float().cpu()-initial_added[key].float()
            parameter_audit.append(dict(name=key, delta_rms=float(delta.square().mean().sqrt()),
                                        delta_max=float(delta.abs().max())))
    save(destination/'parameter_audit.json', dict(parameters=parameter_audit,
         note='Delta from deterministic initialization, not from resumed epoch; gradient maxima in history'))
    probe_modules(args, model, dataset, seed, destination)
    fingerprint = digest(last)
    result = dict(dataset=dataset, method=method, seed=seed, group=name,
                  checkpoint_sha256=fingerprint, added_parameters=sum(p.numel() for p in added),
                  training_objective=method,epochs=EPOCHS,
                  inference='S_RAW except B0; completion heads used only for training',
                  clean_reconstruction_target_used=False)
    if dataset=='private':
        dev = PrivatePaperDataset(args.private_dir, args.private_dir/'paper_split_v1/dev_indices.npy')
        update(args, phase='evaluation', group=name, condition='natural')
        result['metrics'] = evaluate(model, DataLoader(dev, batch_size=1, num_workers=0), args.device, 2)
        details = private_details(model, dev, args.device)
        save(destination/'per_frame.json', details)
        result['metrics'].update({k:v for k,v in details.items() if k!='rows'})
    else:
        scores = []
        for index,(family,severity) in enumerate(CONDITIONS,1):
            condition = f'{family}_{severity}'
            update(args, phase='evaluation', group=name, condition=condition, evaluation=index, evaluations=22)
            path = destination/'curves'/f'{condition}.json'
            row = read(path) if path.exists() else None
            if row is None or row.get('checkpoint_sha256')!=fingerprint:
                row = regional(model, DiagnosticData(args.nyuv2_dir, family, severity), args.device)
                row['checkpoint_sha256'] = fingerprint
                save(path,row)
            if method!='PAIR':
                ref = read(args.output_dir/f'nyuv2/PAIR_seed{seed}/curves/{condition}.json')
                if row['degradation_audit']!=ref['degradation_audit']:
                    raise ValueError('Different condition/sample budget')
            scores.append(dict(condition=condition, miou=row['regions']['all']['miou']))
        result.update(scores=scores, clean_miou=scores[0]['miou'],
                      corruption_mean_miou=float(np.mean([r['miou'] for r in scores[1:]])))
        for condition in ('photon_proxy_heavy','joint_heavy'):
            result[condition+'_miou'] = next(r['miou'] for r in scores if r['condition']==condition)
    result['elapsed_minutes'] = prior_minutes+(time.time()-started)/60
    save(result_path,result)
    print(f'REPLICATION_GROUP_COMPLETE {name} result={json.dumps(result.get("metrics", {"clean":result.get("clean_miou"),"corrupt_mean":result.get("corruption_mean_miou")}))}',flush=True)
    del model, optimizer, scaler
    if amp:
        gc.collect()  # Release the decoder hook's bound-method reference cycle.
        torch.cuda.empty_cache()
    return result


def verify_reference(args, fingerprints):
    """Read-only provenance gate for reusing seed42; no old trained weights needed."""
    root = args.reference_dir
    old = read(root/'protocol.json')
    complete = read(root/'complete.json')
    if (old.get('version')!='paired_v1' or old.get('epochs')!=20
            or complete.get('epochs_per_group')!=20 or complete.get('final_test_access') is not False):
        raise ValueError('Reference must be completed dev-only paired_v1 at20epochs')
    for key, value in old['fingerprints'].items():
        if fingerprints.get(key) != value:
            raise ValueError(f'Reference data/source fingerprint changed: {key}')
    package = Path(__file__).parent.parent
    for key, value in old['code'].items():
        if digest(package/Path(key.replace('\\','/'))) != value:
            raise ValueError(f'Original experiment dependency changed: {key}')
    paths = [root/'protocol.json', root/'complete.json']
    rows = []
    for dataset in ('nyuv2','private'):
        for method in METHODS:
            folder = root/dataset/f'{method}_seed42'
            result_path = folder/'result.json'
            result = read(result_path)
            if (result.get('dataset'),result.get('method'),result.get('seed'),result.get('epochs')) != (dataset,method,42,20):
                raise ValueError(f'Wrong reference result: {folder}')
            history = read(folder/'history.json')['epochs']
            if [r['epoch'] for r in history] != list(range(1,21)):
                raise ValueError('Reference history incomplete')
            paths.extend([result_path,folder/'history.json'])
            if dataset == 'nyuv2':
                expected = [f'{f}_{v}' for f,v in CONDITIONS]
                if [r['condition'] for r in result['scores']] != expected:
                    raise ValueError('Reference curve set incomplete')
                for score in result['scores']:
                    path = folder/'curves'/f'{score["condition"]}.json'
                    curve = read(path)
                    if (curve['checkpoint_sha256'] != result['checkpoint_sha256']
                            or curve['regions']['all']['miou'] != score['miou']):
                        raise ValueError('Reference curve/result disagreement')
                    paths.append(path)
            rows.append(dict(result, provenance='reused_seed42', reference_path=str(result_path)))
    frozen = {str(p.relative_to(root)):digest(p) for p in paths}
    return frozen, rows


def summarize(args, results):
    rows = list(args.reference_results) + list(results)
    summaries, paired = [], []
    for dataset in ('private','nyuv2'):
        keys = (['foreground_iou','boundary_f1_1px','background_fp_frame_rate','tiny_component_recall_iou50_area80']
                if dataset=='private' else ['clean_miou','corruption_mean_miou',
                                           'photon_proxy_heavy_miou','joint_heavy_miou'])
        def value(row,key):
            return (row['metrics'] if dataset=='private' else row)[key]
        candidates = [r for r in rows if r['dataset']==dataset]
        # Only complete PAIR+CONS seed pairs enter aggregate comparisons.
        complete_seeds = sorted(s for s in {r['seed'] for r in candidates}
                                if all(any(r['seed']==s and r['method']==m for r in candidates) for m in METHODS))
        for seed in complete_seeds:
            pair = {m:next(r for r in candidates if r['seed']==seed and r['method']==m) for m in METHODS}
            paired.append(dict(dataset=dataset,seed=seed,comparison='CONS_minus_PAIR',
                delta_pp={k:100*(value(pair['CONS'],k)-value(pair['PAIR'],k))
                          if value(pair['CONS'],k) is not None and value(pair['PAIR'],k) is not None else None for k in keys}))
        for method in METHODS:
            selected = [r for r in candidates if r['method']==method and r['seed'] in complete_seeds]
            summary = dict(dataset=dataset,method=method,seeds=complete_seeds)
            for key in keys:
                values = [value(r,key) for r in selected if value(r,key) is not None]
                deltas = ([p['delta_pp'][key] for p in paired if p['dataset']==dataset and p['delta_pp'][key] is not None]
                          if method=='CONS' else [0.]*len(values))
                summary[key] = dict(mean=float(np.mean(values)) if values else None,
                    std=float(np.std(values,ddof=1)) if len(values)>1 else None,
                    paired_delta_pp=float(np.mean(deltas)) if deltas else None,
                    paired_delta_std_pp=float(np.std(deltas,ddof=1)) if len(deltas)>1 else None)
            summaries.append(summary)
            print('REPLICATION_SUMMARY '+json.dumps(summary),flush=True)
    save(args.output_dir/'comparison.json',dict(groups=rows,summaries=summaries,paired_comparisons=paired,
         new_training_groups=len(results),reused_groups=len(args.reference_results),
         interpretation='Seed42 exploratory selection; seeds777/2025 replication. Mean/sample std, not significance. '
                        'Only complete matched seed pairs enter summaries; final tests untouched.'))


def execute(args):
    args.session_started = time.time()
    update(args,phase='preflight',epochs=EPOCHS)
    protocol = preflight(args)
    previous_minutes = sum(r['elapsed_minutes'] for r in args.reference_results)
    save(args.output_dir/'schedule.json',dict(epochs=20,seeds=list(SEEDS),methods=list(METHODS),
         groups=8,budget_hours=args.budget_hours,historical_estimated_minutes=2*previous_minutes,
         note='Fixed complete matrix; about6h from actual seed42 timing, may exceed soft budget. '
              'No score-based skipping, no early termination; no GPU benchmark needed for unchanged models.'))
    results = []
    summarize(args,results)
    for seed in SEEDS:
        for method in METHODS:
            for dataset in ('nyuv2','private'):
                args.completed = len(results)
                results.append(train_group(args,dataset,method,seed,protocol))
                summarize(args,results)
    save(args.output_dir/'complete.json',dict(training_groups=len(results),planned_seeds=list(SEEDS),
         epochs_per_group=20,public_condition_evaluations=88,private_evaluations=4,
         reused_seed42_groups=4,combined_seeds=[42,*SEEDS],final_test_access=False,
         elapsed_minutes=(time.time()-args.session_started)/60))
    args.completed = len(results)
    update(args,phase='complete',groups_planned=8,seeds=list(SEEDS))
    print('REPLICATION_COMPLETE new_groups=8 combined_seeds=42,777,2025',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    defaults=dict(private_dir='data/new_data/merged',nyuv2_dir='data/public_semseg/nyuv2/processed',
        private_source='data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B1',
        public_source='data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2',
        repair_dir='data/paper_benchmark_runs/repair_window_v1',reference_dir='data/paper_benchmark_runs/paired_window_v1',
        output_dir='data/paper_benchmark_runs/paired_replication_v1')
    for k,v in defaults.items():
        parser.add_argument('--'+k.replace('_','-'),type=Path,default=Path(v))
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--budget-hours',type=float,default=6.)
    args=parser.parse_args()
    if not np.isfinite(args.budget_hours) or args.budget_hours<=0:
        parser.error('budget-hours must be finite and positive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    lock=args.output_dir/'RUNNING.lock'
    with lock.open('x',encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(),host=socket.gethostname()),handle)
    failure=args.output_dir/'failure.json'
    if failure.exists():
        failure.replace(args.output_dir/f'failure.previous.{time.time_ns()}.json')
    try:
        execute(args)
    except BaseException:
        error=traceback.format_exc()
        save(args.output_dir/'failure.json',dict(error=error))
        update(args,phase='failed',error=error)
        raise
    finally:
        # Preserve lock history rather than deleting user-visible files.
        lock.replace(args.output_dir/f'RUNNING.finished.{time.time_ns()}.json')


if __name__=='__main__':
    main()
