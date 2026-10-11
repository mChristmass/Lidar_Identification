"""Two-to-three-hour matched ten-epoch dual-domain context screen, seed42 only."""
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

from .data import PrivatePaperDataset, read_manifest
from .diagnose_phase1 import DiagnosticData, digest, regional
from .context_model import ContextCMX, task_loss
from .run_structure_window import (PrivateTrain, source_path, load_source, audit_inputs,
                                   component_counts, private_details)
from .run import seed_all, evaluate
from .run_noise_window import CONDITIONS
from .train_mixed_control import TrainingData, atomic_checkpoint
from .train_reliability import read, save

METHODS = ('B0', 'S_RAW', 'F_CTX', 'SF_CTX')
EPOCHS = 10


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
        for seed in (42,):
            files[f'source/{dataset}/{seed}'] = source_path(args, dataset, seed)
            source = load_source(args, dataset, seed)
            del source
    fingerprints = {key: digest(path) for key, path in files.items()}
    for key, value in prior['private_sha256'].items():
        if key!='checkpoint' and fingerprints['private/'+key]!=value:
            raise ValueError(f'Private frozen data changed: {key}')
    package = Path(__file__).parent
    code = sorted(package.glob('*.py'))+sorted((package.parent/'cmx_initial_transfer/official').glob('*.py'))
    protocol = dict(version='context_v1', fingerprints=fingerprints,
                    code={str(p.relative_to(package.parent)): digest(p) for p in code},
                    epochs=EPOCHS, methods=list(METHODS), seeds=[42], budget_hours=args.budget_hours,
                    scope='dev-only broad screen; BOTH datasets participate; no efficacy gate',
                    module_design='B0; S_RAW; F_CTX top-down feature residual; SF_CTX both branches',
                    clean_reconstruction_target_used=False,
                    sources='private best B1 / public clean best B2, identical per dataset/seed across methods',
                    training='private natural flip; public mixed v2; CE only ALL methods; fixed final10',
                    optimizer='AdamW base1e-5 added1e-3 wd.01 poly.9 clip1; batch8 private / batch2 accum4 public',
                    observed_inputs=['intensity','depth','validity'], interpretation='context feature correction, not image denoising',
                    schedule='all eight groups at ten epochs; one seed; budget is soft, no score-based pruning')
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
                if getattr(model,'context_method',None) in ('F_CTX','SF_CTX'):
                    model.adapters_enabled=True
                    model.context_enabled=False
                    context_off=model(x)['logits'].float()
                    row['context_only_off_logit_change']=float((on-context_off).abs().mean())
                    row['context_only_off_prediction_change']=float((on.argmax(1)!=context_off.argmax(1)).float().mean())
                rows.append(row)
            finally:
                model.adapters_enabled = True
                model.audit_enabled = False
                model.context_enabled = True
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
    model = ContextCMX(method, source['args']['model'], 2 if dataset=='private' else 40,
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
        steps, skipped_steps, added_grad_max = 0, 0, 0.
        for index, (x,y) in enumerate(loader):
            x, y = x.to(args.device), y.to(args.device)
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr']*(1-((epoch-1)+index/len(loader))/EPOCHS)**.9
            window_samples = min(batch*accum, len(data)-(index//accum)*batch*accum)
            with torch.autocast(device_type='cuda', enabled=amp):
                ce = task_loss(model(x), y)
                loss = ce*len(x)/window_samples
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{name}/{epoch}/{index}')
            scaler.scale(loss).backward()
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
            loss_sum += float(ce.detach())*len(x)
            if index%25==0 or index+1==len(loader):
                update(args, phase='training', group=name, epoch=epoch, epochs=EPOCHS,
                       batch=index+1, batches=len(loader), loss=loss_sum/min((index+1)*batch,len(data)))
        history.append(dict(epoch=epoch, loss=loss_sum/len(data), added_grad_max=added_grad_max,
                            attempted_steps=steps, skipped_amp_steps=skipped_steps))
        if steps == skipped_steps:
            raise RuntimeError('All optimizer steps skipped in this epoch')
        atomic_checkpoint(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scaler=scaler.state_dict(), history=history, epoch=epoch, protocol=protocol, group=name,
                          elapsed_minutes=prior_minutes+(time.time()-started)/60))
        save(destination/'history.json', dict(epochs=history))
        print(f'CONTEXT_EPOCH {name} epoch={epoch}/{EPOCHS} loss={history[-1]["loss"]:.6f}', flush=True)
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
                  training_objective='CE',epochs=EPOCHS,
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
            if method!='B0':
                ref = read(args.output_dir/f'nyuv2/B0_seed{seed}/curves/{condition}.json')
                if row['degradation_audit']!=ref['degradation_audit']:
                    raise ValueError('Different condition/sample budget')
            scores.append(dict(condition=condition, miou=row['regions']['all']['miou']))
        result.update(scores=scores, clean_miou=scores[0]['miou'],
                      corruption_mean_miou=float(np.mean([r['miou'] for r in scores[1:]])))
        for condition in ('photon_proxy_heavy','joint_heavy'):
            result[condition+'_miou'] = next(r['miou'] for r in scores if r['condition']==condition)
    result['elapsed_minutes'] = prior_minutes+(time.time()-started)/60
    save(result_path,result)
    print(f'CONTEXT_GROUP_COMPLETE {name} result={json.dumps(result.get("metrics", {"clean":result.get("clean_miou"),"corrupt_mean":result.get("corruption_mean_miou")}))}',flush=True)
    del model, optimizer, scaler
    if amp:
        torch.cuda.empty_cache()
    return result


def summarize(args, results):
    summaries = []
    for dataset in ('private','nyuv2'):
        for method in METHODS:
            rows = [r for r in results if r['dataset']==dataset and r['method']==method]
            if not rows:
                continue
            metrics = (['foreground_iou','boundary_f1_1px','background_fp_frame_rate','tiny_component_recall_iou50_area80']
                       if dataset=='private' else ['clean_miou','corruption_mean_miou',
                                                   'photon_proxy_heavy_miou','joint_heavy_miou'])
            summary = dict(dataset=dataset,method=method,seeds=[r['seed'] for r in rows])
            for key in metrics:
                values, gains = [], []
                for r in rows:
                    base = next(b for b in results if b['dataset']==dataset and b['method']=='B0' and b['seed']==r['seed'])
                    value = (r['metrics'] if dataset=='private' else r)[key]
                    reference = (base['metrics'] if dataset=='private' else base)[key]
                    if value is not None and reference is not None:
                        values.append(value); gains.append(100*(value-reference))
                summary[key] = dict(mean=float(np.mean(values)) if values else None,
                     std=float(np.std(values,ddof=1)) if len(values)>1 else None,
                     paired_delta_pp=float(np.mean(gains)) if gains else None)
            summaries.append(summary)
            print('CONTEXT_SUMMARY '+json.dumps(summary),flush=True)
    paired_comparisons = []
    for dataset in ('private','nyuv2'):
        pair={r['method']:r for r in results if r['dataset']==dataset and r['seed']==42}
        keys=['foreground_iou','boundary_f1_1px','background_fp_frame_rate'] if dataset=='private' else [
            'clean_miou','corruption_mean_miou','photon_proxy_heavy_miou','joint_heavy_miou']
        def metrics(method):
            return pair[method]['metrics'] if dataset=='private' else pair[method]
        for name,left,right in [('F_CTX_minus_B0','F_CTX','B0'),
                                ('SF_CTX_minus_S_RAW','SF_CTX','S_RAW')]:
            if left in pair and right in pair:
                paired_comparisons.append(dict(dataset=dataset,comparison=name,seed=42,
                    delta_pp={k:100*(metrics(left)[k]-metrics(right)[k]) for k in keys}))
        if all(m in pair for m in METHODS):
            paired_comparisons.append(dict(dataset=dataset,comparison='factorial_interaction',seed=42,
                delta_pp={k:100*(metrics('SF_CTX')[k]-metrics('S_RAW')[k]-metrics('F_CTX')[k]+metrics('B0')[k]) for k in keys},
                note='Descriptive difference of differences; one seed, not significance or mechanism proof'))
    save(args.output_dir/'comparison.json',dict(groups=results,summaries=summaries,
         paired_comparisons=paired_comparisons,
         interpretation='Single-seed 10-epoch screen; separate domains; std null; compare ONLY matched current controls'))


def device_smoke(args):
    """Server-only synthetic shape/AMP/memory check before committing to hours of work."""
    if not args.device.startswith('cuda'):
        return
    rows = []
    for dataset,variant,classes,h,w,batch in (('private','CMX_B1',2,128,128,8),
                                             ('nyuv2','CMX_B2',40,480,640,2)):
        for method in ('F_CTX','SF_CTX'):
            update(args,phase='device_smoke',group=f'{dataset}/{method}')
            seed_all(42)
            model = ContextCMX(method,variant,classes,max(h,w),256).to(args.device).train()
            optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5)
            torch.cuda.reset_peak_memory_stats()
            x = torch.rand(batch,3,h,w,device=args.device)
            x[:,2]=(x[:,2]>.3).float()
            y = torch.randint(classes,(batch,h,w),device=args.device)
            with torch.autocast(device_type='cuda'):
                loss = task_loss(model(x),y)
            loss.backward()
            if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError(f'Non-finite synthetic smoke: {dataset}/{method}')
            optimizer.step()  # Allocate Adam states too; forward-only memory would underestimate training.
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad():
                check = model(x[:1])['logits']
                if not torch.isfinite(check).all():
                    raise FloatingPointError('Non-finite FP32 evaluation smoke')
            torch.cuda.synchronize()
            rows.append(dict(dataset=dataset,method=method,loss=float(loss.detach()),
                             peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2))
            del model,optimizer,x,y,loss,check
            torch.cuda.empty_cache()
    save(args.output_dir/'device_smoke.json',dict(rows=rows,
         note='One synthetic Adam step plus FP32 evaluation per shape; no real samples; not a runtime guarantee'))


def execute(args):
    started = time.time()
    args.session_started = started
    update(args,phase='preflight')
    protocol = preflight(args)
    device_smoke(args)
    audit_inputs(args)
    results = []
    for method in METHODS:
        for dataset in ('nyuv2','private'):
            args.completed=len(results)
            results.append(train_group(args,dataset,method,42,protocol))
            summarize(args,results)
    seeds = sorted({r['seed'] for r in results})
    save(args.output_dir/'complete.json',dict(training_groups=len(results), planned_seeds=seeds,
         groups_by_seed={str(seed):sum(r['seed']==seed for r in results) for seed in seeds},
         public_condition_evaluations=22*sum(r['dataset']=='nyuv2' for r in results),
         private_evaluations=sum(r['dataset']=='private' for r in results),
         elapsed_minutes=(time.time()-started)/60, final_test_access=False,
         epochs_per_group=EPOCHS,note='All eight matched 10-epoch groups; do not compare absolute scores with 20-epoch runs'))
    args.completed=len(results)
    update(args,phase='complete',groups_planned=len(results),seeds=seeds)
    print(f'CONTEXT_COMPLETE groups={len(results)} seeds={seeds}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    defaults=dict(private_dir='data/new_data/merged',nyuv2_dir='data/public_semseg/nyuv2/processed',
        private_source='data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B1',
        public_source='data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2',
        repair_dir='data/paper_benchmark_runs/repair_window_v1',output_dir='data/paper_benchmark_runs/context_window_v1')
    for k,v in defaults.items():
        parser.add_argument('--'+k.replace('_','-'),type=Path,default=Path(v))
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--budget-hours',type=float,default=3.)
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
