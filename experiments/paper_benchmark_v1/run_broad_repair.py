"""Ten-hour broad screen: three repair/detail candidates, without Mamba."""
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
from .broad_window_model import BroadCMX as PairedCMX, backward_objective
from .run_paired_replication import verify_reference as verify_seed42
from .paired_model import VIEW_CONFIG
from .run_structure_window import (PrivateTrain, source_path, load_source, audit_inputs,
                                   component_counts, private_details)
from .run import seed_all, evaluate
from .run_noise_window import CONDITIONS
from .train_mixed_control import TrainingData, atomic_checkpoint
from .train_reliability import read, save

METHODS = ('R_RES', 'R_FILTER', 'F_DETAIL')
SEEDS = (42, 777)
EPOCHS = 20


def update(args, **state):
    if hasattr(args, 'session_started'):
        state['elapsed_minutes'] = round((time.time()-args.session_started)/60, 1)
        state['budget_remaining_minutes'] = round(max(0.,args.budget_hours*60-state['elapsed_minutes']),1)
    save(args.output_dir/'status.json', dict(updated_unix=time.time(),
         groups_completed=getattr(args, 'completed', 0), groups_possible=12, **state))


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
        for seed in (42, 777, 2025):
            files[f'source/{dataset}/{seed}'] = source_path(args, dataset, seed)
            source = load_source(args, dataset, seed)
            del source
    fingerprints = {key: digest(path) for key, path in files.items()}
    for key, value in prior['private_sha256'].items():
        if key!='checkpoint' and fingerprints['private/'+key]!=value:
            raise ValueError(f'Private frozen data changed: {key}')
    package = Path(__file__).parent
    # Abandoned local drafts are not dependencies and must not be required on the server.
    excluded = {'mamba_pilot.py', 'run_broad_mamba.py'}
    code = sorted(p for p in package.glob('*.py') if p.name not in excluded)
    code += sorted((package.parent/'cmx_initial_transfer/official').glob('*.py'))
    reference_files, args.reference_results = verify_reference(args, fingerprints)
    protocol = dict(version='broad_repair_v1', fingerprints=fingerprints,
                    code={str(p.relative_to(package.parent)): digest(p) for p in code},
                    epochs=EPOCHS, methods=list(METHODS), seeds=list(SEEDS), budget_hours=args.budget_hours,
                    scope='dev-only broad screen; BOTH datasets participate; no efficacy gate',
                    module_design='PAIR plus R_RES task correction / R_FILTER convex smoothing / F_DETAIL high-resolution refinement',
                    candidate_rule='repair regularizer .01; no KL',
                    reference_files=reference_files, view_config=VIEW_CONFIG,
                    consistency_weight=0., completion_weight=0., correction_weight=.01,
                    clean_reconstruction_target_used=False,
                    sources='private best B1 / public clean best B2, identical per dataset/seed across methods',
                    training='reference private natural flip/public mixed v2; paired same extra noise and block depth masking; fixed final epoch',
                    optimizer='AdamW base1e-5 added1e-3 wd.01 poly.9 clip1; batch8 private / batch2 accum4 public',
                    observed_inputs=['intensity','depth','validity'], interpretation='observed semantic/feature targets, not clean image or physical depth reconstruction',
                    schedule='fixed12 groups; three candidates seeds42/777 both domains; soft10h')
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
                model.candidate_enabled = True
                model.audit_enabled = True
                on = model(x)['logits'].float()
                stages = model.audit_rows
                model.candidate_enabled = False
                off = model(x)['logits'].float()
                row=dict(regime=regime,train_index=int(index),stages=stages,
                    logit_mean_abs_change=float((on-off).abs().mean()),
                    prediction_changed_fraction=float((on.argmax(1)!=off.argmax(1)).float().mean()))
                rows.append(row)
            finally:
                model.adapters_enabled = True
                model.candidate_enabled = True
                model.audit_enabled = False
                model.capture_features = False
    save(destination/'module_probe.json',dict(rows=rows,
         note='Candidate-only toggle, retaining learned S_RAW. Six train samples/regime; not full-dev or causal ablation.'))

def train_group(args, dataset, method, seed, protocol):
    name = f'{dataset}/{method}_seed{seed}'
    destination = args.output_dir/name
    destination.mkdir(parents=True, exist_ok=True)
    last, result_path = destination/'last.pth', destination/'result.json'
    if result_path.exists():
        result = read(result_path)
        if (result.get('dataset'),result.get('method'),result.get('seed'),result.get('epochs')) != (dataset,method,seed,EPOCHS):
            raise ValueError(f'Completed result identity mismatch: {name}')
        if [r['epoch'] for r in read(destination/'history.json')['epochs']] != list(range(1,EPOCHS+1)):
            raise ValueError(f'Completed history incomplete: {name}')
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
        candidate_grad_max = 0.
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
                    candidate_grad = max((p.grad.detach().float().abs().max().item()
                        for k,p in model.named_parameters() if k.startswith('adapters.candidate.')
                        and p.grad is not None), default=0.)
                    if np.isfinite(candidate_grad):
                        candidate_grad_max = max(candidate_grad_max,candidate_grad)
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
                            candidate_grad_max=candidate_grad_max,
                            attempted_steps=steps, skipped_amp_steps=skipped_steps))
        if steps == skipped_steps:
            raise RuntimeError('All optimizer steps skipped in this epoch')
        atomic_checkpoint(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scaler=scaler.state_dict(), history=history, epoch=epoch, protocol=protocol, group=name,
                          elapsed_minutes=prior_minutes+(time.time()-started)/60))
        save(destination/'history.json', dict(epochs=history))
        if epoch % 5 == 0 or epoch == 1:
            print(f'BROAD_EPOCH {name} epoch={epoch}/{EPOCHS} loss={history[-1]["loss"]:.6f}', flush=True)
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
                  inference='S_RAW plus candidate; all candidate modules remain active at inference',
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
            if method in METHODS:
                ref_root = args.reference_dir if seed==42 else args.replication_dir
                ref = read(ref_root/f'nyuv2/PAIR_seed{seed}/curves/{condition}.json')
                if row['degradation_audit']!=ref['degradation_audit']:
                    raise ValueError('Different condition/sample budget')
            scores.append(dict(condition=condition, miou=row['regions']['all']['miou']))
        result.update(scores=scores, clean_miou=scores[0]['miou'],
                      corruption_mean_miou=float(np.mean([r['miou'] for r in scores[1:]])))
        for condition in ('photon_proxy_heavy','joint_heavy'):
            result[condition+'_miou'] = next(r['miou'] for r in scores if r['condition']==condition)
    result['elapsed_minutes'] = prior_minutes+(time.time()-started)/60
    save(result_path,result)
    print(f'BROAD_GROUP_COMPLETE {name} result={json.dumps(result.get("metrics", {"clean":result.get("clean_miou"),"corrupt_mean":result.get("corruption_mean_miou")}))}',flush=True)
    del model, optimizer, scaler
    if amp:
        gc.collect()  # Release the decoder hook's bound-method reference cycle.
        torch.cuda.empty_cache()
    return result


def verify_reference(args, fingerprints):
    frozen, rows = verify_seed42(args,fingerprints)
    frozen = {'paired/'+k:v for k,v in frozen.items()}
    package = Path(__file__).parent.parent
    for label,root,version,methods,seeds,expected_groups in (
        ('replication',args.replication_dir,'paired_replication_v1',('PAIR',),(777,),8),):
        protocol = read(root/'protocol.json')
        complete = read(root/'complete.json')
        if (protocol.get('version')!=version or protocol.get('epochs')!=20
                or complete.get('training_groups')!=expected_groups
                or complete.get('epochs_per_group')!=20 or complete.get('final_test_access') is not False):
            raise ValueError(f'Wrong historical protocol: {label}')
        for key,value in protocol['fingerprints'].items():
            if fingerprints.get(key)!=value:
                raise ValueError(f'Historical data/source changed: {label}/{key}')
        for key,value in protocol['code'].items():
            if digest(package/Path(key.replace('\\','/')))!=value:
                raise ValueError(f'Historical code changed: {key}')
        paths = [root/'protocol.json',root/'complete.json']
        for dataset in ('nyuv2','private'):
            for seed in seeds:
                for method in methods:
                    folder = root/dataset/f'{method}_seed{seed}'
                    result = read(folder/'result.json')
                    if (result.get('dataset'),result.get('method'),result.get('seed'),result.get('epochs')) != (dataset,method,seed,20):
                        raise ValueError('Wrong historical result')
                    if [r['epoch'] for r in read(folder/'history.json')['epochs']]!=list(range(1,21)):
                        raise ValueError('Incomplete historical epochs')
                    paths.extend([folder/'result.json',folder/'history.json'])
                    if dataset=='nyuv2':
                        if [r['condition'] for r in result['scores']]!=[f'{f}_{v}' for f,v in CONDITIONS]:
                            raise ValueError('Incomplete historical conditions')
                        for score in result['scores']:
                            path = folder/'curves'/f'{score["condition"]}.json'
                            curve = read(path)
                            if curve['checkpoint_sha256']!=result['checkpoint_sha256'] or curve['regions']['all']['miou']!=score['miou']:
                                raise ValueError('Historical curve/result disagreement')
                            paths.append(path)
                    rows.append(dict(result,provenance='reused_'+label,reference_path=str(folder/'result.json')))
        frozen.update({label+'/'+str(p.relative_to(root)):digest(p) for p in paths})
    return frozen, rows


def summarize(args, results):
    rows = list(args.reference_results)+list(results)
    comparisons, summaries = [], []
    for dataset in ('private','nyuv2'):
        keys = (['foreground_iou','boundary_f1_1px','background_fp_frame_rate','tiny_component_recall_iou50_area80']
                if dataset=='private' else ['clean_miou','corruption_mean_miou',
                                           'photon_proxy_heavy_miou','joint_heavy_miou'])
        def metric(row,k):
            return (row['metrics'] if dataset=='private' else row)[k]
        for method,reference in ((m,'PAIR') for m in METHODS):
            seed_rows = []
            for seed in SEEDS:
                pair = {r['method']:r for r in rows if r['dataset']==dataset and r['seed']==seed}
                if method not in pair or reference not in pair:
                    continue
                item = dict(dataset=dataset,method=method,reference=reference,seed=seed,
                    delta_pp={k:100*(metric(pair[method],k)-metric(pair[reference],k))
                              if metric(pair[method],k) is not None and metric(pair[reference],k) is not None else None for k in keys})
                comparisons.append(item);seed_rows.append(item)
            summary = dict(dataset=dataset,method=method,reference=reference,seeds=[r['seed'] for r in seed_rows])
            if not seed_rows:
                continue
            for k in keys:
                values = [r['delta_pp'][k] for r in seed_rows if r['delta_pp'][k] is not None]
                actual = [metric(r,k) for r in rows if r['dataset']==dataset and r['method']==method
                          and r['seed'] in summary['seeds'] and metric(r,k) is not None]
                summary[k] = dict(mean_delta_pp=float(np.mean(values)) if values else None,
                    std_delta_pp=float(np.std(values,ddof=1)) if len(values)>1 else None,
                    mean=float(np.mean(actual)) if actual else None,
                    std=float(np.std(actual,ddof=1)) if len(actual)>1 else None)
            summaries.append(summary)
            print('BROAD_SUMMARY '+json.dumps(summary),flush=True)
    save(args.output_dir/'comparison.json',dict(groups=rows,paired_comparisons=comparisons,summaries=summaries,
         new_training_groups=len(results),reused_groups=len(args.reference_results),
         interpretation='Pairwise summaries use only shared seeds. Seed42 selected during exploration; '
                        'All three candidates use seeds42/777; not novelty or significance proof.'))


def execute(args):
    args.session_started=time.time()
    update(args,phase='preflight',epochs=20)
    protocol=preflight(args)
    if getattr(args,'preflight_only',False):
        update(args,phase='preflight_complete')
        print('BROAD_PREFLIGHT_OK training_not_started',flush=True)
        return
    cost=sum(r['elapsed_minutes'] for r in args.reference_results if r['method']=='PAIR' and r['seed']==42)
    save(args.output_dir/'schedule.json',dict(epochs=20,seeds=list(SEEDS),groups=12,budget_hours=args.budget_hours,
         historical_estimated_minutes=6*cost,note='Fixed complete matrix; soft budget, no score pruning or shortened training.'))
    results=[]
    summarize(args,results)
    for seed in SEEDS:
        for method in METHODS:
            for dataset in ('nyuv2','private'):
                args.completed=len(results)
                results.append(train_group(args,dataset,method,seed,protocol))
                args.completed=len(results)
                update(args,phase='group_complete',group=results[-1]['group'])
                summarize(args,results)
    save(args.output_dir/'complete.json',dict(training_groups=12,planned_seeds=list(SEEDS),epochs_per_group=20,
         public_condition_evaluations=132,private_evaluations=6,reused_groups=len(args.reference_results),
         final_test_access=False,elapsed_minutes=(time.time()-args.session_started)/60))
    args.completed=12
    update(args,phase='complete',groups_planned=12,seeds=list(SEEDS))
    print('BROAD_COMPLETE new_groups=12 seeds=42,777',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    defaults=dict(private_dir='data/new_data/merged',nyuv2_dir='data/public_semseg/nyuv2/processed',
        private_source='data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B1',
        public_source='data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2',
        repair_dir='data/paper_benchmark_runs/repair_window_v1',reference_dir='data/paper_benchmark_runs/paired_window_v1',
        replication_dir='data/paper_benchmark_runs/paired_replication_v1',
        output_dir='data/paper_benchmark_runs/broad_repair_v1')
    for k,v in defaults.items():
        parser.add_argument('--'+k.replace('_','-'),type=Path,default=Path(v))
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--budget-hours',type=float,default=10.)
    parser.add_argument('--preflight-only',action='store_true',help='Validate and freeze protocol without training')
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
