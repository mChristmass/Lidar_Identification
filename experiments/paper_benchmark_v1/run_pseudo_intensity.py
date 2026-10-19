"""Two-seed, two-domain pseudo-intensity experiment; frozen dev only."""
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

from .data import PrivatePaperDataset
from .diagnose_phase1 import DiagnosticData, digest, regional
from .pseudo_intensity_model import PseudoIntensityCMX, objective, PSEUDO_WEIGHT
from .run import seed_all, evaluate
from .run_noise_window import CONDITIONS
from .run_structure_window import PrivateTrain, private_details, load_source, source_path
from .task_control_model import CORRECTION_WEIGHT
from .train_mixed_control import TrainingData, atomic_checkpoint
from .train_reliability import read, save

SEEDS = (42, 777)
METHODS = ('FIXED', 'PSEUDO_SUP')
EPOCHS = 20
GROUPS = 8


def status(args, **items):
    save(args.output_dir/'status.json', dict(updated_unix=time.time(),
         elapsed_minutes=round((time.time()-args.started)/60, 1),
         groups_completed=args.completed, groups_planned=GROUPS, **items))


def reference(args, dataset, seed, method):
    folder = args.task_dir/dataset/f'{method}_seed{seed}'
    result = read(folder/'result.json')
    history = read(folder/'history.json')['epochs']
    if (result['dataset'], result['method'], result['seed']) != (dataset, method, seed):
        raise ValueError(f'Wrong reference identity: {folder}')
    if [r['epoch'] for r in history] != list(range(1, EPOCHS+1)):
        raise ValueError(f'Incomplete reference: {folder}')
    if result['checkpoint_sha256'] != digest(folder/'last.pth'):
        raise ValueError(f'Changed reference checkpoint: {folder}')
    if dataset == 'nyuv2' and [r['condition'] for r in result['scores']] != [f'{f}_{s}' for f,s in CONDITIONS]:
        raise ValueError(f'Incomplete reference conditions: {folder}')
    return result


def preflight(args):
    split = args.private_dir/'paper_split_v1'
    train, dev = [np.load(split/f'{part}_indices.npy') for part in ('train','dev')]
    if len(train) != 970 or len(dev) != 137 or np.intersect1d(train,dev).size:
        raise ValueError('Private frozen split differs')
    prior=read(args.task_dir/'protocol.json')
    if prior.get('version')!='task_control_v1' or prior.get('epochs')!=EPOCHS or prior.get('seeds')!=list(SEEDS):
        raise ValueError('Historical task-control protocol differs')
    refs = [reference(args,d,s,m) for s in SEEDS for d in ('nyuv2','private')
            for m in ('B0','R_TASK')]
    sources = {f'{d}/{s}':digest(source_path(args,d,s))
               for s in SEEDS for d in ('nyuv2','private')}
    for d in ('nyuv2','private'):
        for s in SEEDS:
            load_source(args,d,s)
    manifests = {part:digest(args.nyuv2_dir/'manifests'/f'{part}.jsonl') for part in ('train','dev')}
    frozen=prior['fingerprints']
    observed={**{f'source/{key}':value for key,value in sources.items()},
              **{f'nyuv2/{key}_manifest':value for key,value in manifests.items()},
              'private/paper_split_v1/train_indices.npy':digest(split/'train_indices.npy'),
              'private/paper_split_v1/dev_indices.npy':digest(split/'dev_indices.npy')}
    for key,value in observed.items():
        if frozen.get(key)!=value:
            raise ValueError(f'Frozen reference input changed: {key}')
    protocol = dict(version='pseudo_intensity_v1', seeds=list(SEEDS), methods=list(METHODS),
                    epochs=EPOCHS, source_sha256=sources, manifest_sha256=manifests,
                    private_train_indices_sha256=digest(split/'train_indices.npy'),
                    private_dev_indices_sha256=digest(split/'dev_indices.npy'),
                    references={f'{r["dataset"]}/{r["method"]}/{r["seed"]}':r['checkpoint_sha256'] for r in refs},
                    code_sha256={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('pseudo_intensity_model.py'))},
                    objective=dict(correction_l2_weight=CORRECTION_WEIGHT,pseudo_mse_weight=PSEUDO_WEIGHT),
                    pseudo_target='observed intensity -> 3x3 binomial, 50% blend; no actual clean image',
                    training='private natural flip; public mixed v2; final epoch, 20 epochs, matched sources',
                    scope='train/dev only; no final test; references are completed B0 and R_TASK')
    path=args.output_dir/'protocol.json'
    if path.exists() and read(path)!=protocol:
        raise ValueError('Protocol changed: use a fresh output directory')
    save(path,protocol)
    TrainingData(args.nyuv2_dir,'mixed',42)
    return protocol,refs


def train_group(args,dataset,method,seed,protocol):
    name=f'{dataset}/{method}_seed{seed}'
    dest=args.output_dir/name
    dest.mkdir(parents=True,exist_ok=True)
    checkpoint,result_file=dest/'last.pth',dest/'result.json'
    if result_file.exists():
        result=read(result_file)
        if result['checkpoint_sha256']!=digest(checkpoint) or result['group']!=name:
            raise ValueError(f'Completed result differs: {name}')
        print(f'PSEUDO_RESUME_COMPLETE {name}',flush=True)
        return result
    started=time.time()
    seed_all(seed)
    source=load_source(args,dataset,seed)
    model=PseudoIntensityCMX(method,source['args']['model'],2 if dataset=='private' else 40,
                             128 if dataset=='private' else 640,source['args']['decoder_dim']).to(args.device)
    model.load_baseline(source['model'])
    del source
    base,added=[],[]
    for key,p in model.named_parameters():
        (added if key.startswith('adapters.') else base).append(p)
    groups=[dict(params=base,lr=1e-5,initial_lr=1e-5)]
    if added:
        groups.append(dict(params=added,lr=1e-3,initial_lr=1e-3))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    amp=args.device.startswith('cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    start,history,prior_minutes=1,[],0.
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['protocol']!=protocol or state['group']!=name:
            raise ValueError(f'Resume mismatch: {name}')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start,history=state['epoch']+1,state['history']
        prior_minutes=state.get('elapsed_minutes',0.)
        del state
    data=PrivateTrain(args.private_dir,seed) if dataset=='private' else TrainingData(args.nyuv2_dir,'mixed',seed)
    batch,accum=(8,1) if dataset=='private' else (2,4)
    for epoch in range(start,EPOCHS+1):
        data.epoch=epoch
        seed_all(seed+epoch)
        loader=DataLoader(data,batch_size=batch,shuffle=True,num_workers=0,pin_memory=False,
                          generator=torch.Generator().manual_seed(seed+epoch))
        model.train();optimizer.zero_grad(set_to_none=True)
        totals=dict(loss=0.,ce=0.,correction_l2=0.,pseudo_mse=0.)
        steps=skipped=0
        for index,(x,y) in enumerate(loader):
            x,y=x.to(args.device),y.to(args.device)
            for group in optimizer.param_groups:
                group['lr']=group['initial_lr']*(1-((epoch-1)+index/len(loader))/EPOCHS)**.9
            window=min(batch*accum,len(data)-(index//accum)*batch*accum)
            with torch.autocast(device_type='cuda',enabled=amp):
                prediction=model(x)
                loss,parts=objective(prediction,y,method)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: {name}/{epoch}/{index}')
            scaler.scale(loss*len(x)/window).backward()
            if (index+1)%accum==0 or index+1==len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                old=scaler.get_scale()
                scaler.step(optimizer);scaler.update()
                skipped+=int(scaler.get_scale()<old)
                steps+=1
                optimizer.zero_grad(set_to_none=True)
            for key,value in dict(loss=float(loss.detach()),**parts).items():
                totals[key]+=value*len(x)
            if index%25==0 or index+1==len(loader):
                status(args,phase='training',group=name,epoch=epoch,batch=index+1,batches=len(loader),
                       loss=totals['loss']/min((index+1)*batch,len(data)))
        if steps==skipped:
            raise RuntimeError(f'All optimizer steps skipped: {name}/{epoch}')
        row=dict(epoch=epoch,**{k:v/len(data) for k,v in totals.items()},
                 attempted_steps=steps,skipped_amp_steps=skipped)
        history.append(row)
        atomic_checkpoint(checkpoint,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                          scaler=scaler.state_dict(),history=history,epoch=epoch,protocol=protocol,
                          group=name,elapsed_minutes=prior_minutes+(time.time()-started)/60))
        save(dest/'history.json',dict(epochs=history))
        if epoch==1 or epoch%5==0:
            print(f'PSEUDO_EPOCH {name} completed={args.completed}/{GROUPS} {epoch}/{EPOCHS} loss={row["loss"]:.6f} '
                  f'ce={row["ce"]:.6f} pseudo_mse={row["pseudo_mse"]:.6f}',flush=True)
    fingerprint=digest(checkpoint)
    result=dict(group=name,dataset=dataset,method=method,seed=seed,epochs=EPOCHS,
                checkpoint_sha256=fingerprint,added_parameters=sum(p.numel() for p in added),
                pseudo_target='fixed smoothing of observed input; not actual clean data')
    model.eval()
    if dataset=='private':
        dev=PrivatePaperDataset(args.private_dir,args.private_dir/'paper_split_v1/dev_indices.npy')
        status(args,phase='evaluation',group=name,condition='natural')
        result['metrics']=evaluate(model,DataLoader(dev,batch_size=1,num_workers=0,pin_memory=False),args.device,2)
        details=private_details(model,dev,args.device)
        save(dest/'per_frame.json',details)
        result['metrics'].update({k:v for k,v in details.items() if k!='rows'})
    else:
        scores=[]
        for i,(family,severity) in enumerate(CONDITIONS,1):
            condition=f'{family}_{severity}'
            status(args,phase='evaluation',group=name,condition=condition,evaluation=i,evaluations=len(CONDITIONS))
            curve_path=dest/'curves'/f'{condition}.json'
            curve=read(curve_path) if curve_path.exists() else None
            if curve is None or curve.get('checkpoint_sha256')!=fingerprint:
                curve=regional(model,DiagnosticData(args.nyuv2_dir,family,severity),args.device)
                curve['checkpoint_sha256']=fingerprint
                save(curve_path,curve)
            reference_curve=read(args.task_dir/f'nyuv2/B0_seed{seed}/curves/{condition}.json')
            if curve['degradation_audit']!=reference_curve['degradation_audit']:
                raise ValueError(f'Different public corruption samples: {name}/{condition}')
            scores.append(dict(condition=condition,miou=curve['regions']['all']['miou']))
            if i in (1,10,20,len(CONDITIONS)):
                print(f'PSEUDO_EVAL {name} condition={i}/{len(CONDITIONS)} {condition} '
                      f'miou={scores[-1]["miou"]:.4f}',flush=True)
        result.update(scores=scores,clean_miou=scores[0]['miou'],
                      corruption_mean_miou=float(np.mean([r['miou'] for r in scores[1:]])))
        for condition in ('photon_proxy_heavy','joint_heavy'):
            result[condition+'_miou']=next(r['miou'] for r in scores if r['condition']==condition)
    result['elapsed_minutes']=prior_minutes+(time.time()-started)/60
    save(result_file,result)
    metric=result['metrics']['foreground_iou'] if dataset=='private' else result['corruption_mean_miou']
    print(f'PSEUDO_GROUP_COMPLETE {name} completed={args.completed+1}/{GROUPS} '
          f'primary={metric:.4f} elapsed_min={result["elapsed_minutes"]:.1f}',flush=True)
    del model,optimizer,scaler
    if amp:
        gc.collect();torch.cuda.empty_cache()
    return result


def summarize(args,results):
    all_rows=list(args.references)+list(results)
    summaries=[]
    for dataset in ('private','nyuv2'):
        keys=(('foreground_iou','boundary_f1_1px','tiny_component_recall_iou50_area80')
              if dataset=='private' else ('clean_miou','corruption_mean_miou','photon_proxy_heavy_miou','joint_heavy_miou'))
        for method in ('B0','R_TASK',*METHODS):
            rows=[r for r in all_rows if r['dataset']==dataset and r['method']==method]
            if not rows:
                continue
            summary=dict(dataset=dataset,method=method,seeds=[r['seed'] for r in rows],completed=len(rows),planned=2)
            for key in keys:
                values=[];deltas=[]
                for row in rows:
                    value=(row['metrics'] if dataset=='private' else row).get(key)
                    base=next(b for b in all_rows if b['dataset']==dataset and b['method']=='B0' and b['seed']==row['seed'])
                    baseline=(base['metrics'] if dataset=='private' else base).get(key)
                    if value is not None and baseline is not None:
                        values.append(value);deltas.append(100*(value-baseline))
                summary[key]=dict(mean=float(np.mean(values)) if values else None,
                                  sd=float(np.std(values,ddof=1)) if len(values)>1 else None,
                                  delta_B0_pp=float(np.mean(deltas)) if deltas else None)
            summaries.append(summary)
            if method in METHODS:
                key=keys[0] if dataset=='private' else 'corruption_mean_miou'
                print(f'PSEUDO_GROUP_MEAN {dataset}/{method} {len(rows)}/2 '
                      f'{key}={summary[key]["mean"]:.4f} delta_B0_pp={summary[key]["delta_B0_pp"]:+.2f}',flush=True)
    save(args.output_dir/'comparison.json',dict(groups=all_rows,summaries=summaries,
         interpretation='Paired seed comparisons. Pseudo target is only a fixed-smoothed observation; '
                        'a gain does not imply recovery of physically clean intensity.'))


def execute(args):
    args.started=time.time();args.completed=0
    status(args,phase='preflight')
    protocol,args.references=preflight(args)
    if args.preflight_only:
        print('PSEUDO_PREFLIGHT_OK',flush=True);return
    results=[]
    for seed in SEEDS:
        for method in METHODS:
            for dataset in ('nyuv2','private'):
                print(f'PSEUDO_START {dataset}/{method}_seed{seed} completed={args.completed}/{GROUPS}',flush=True)
                result=train_group(args,dataset,method,seed,protocol)
                results.append(result);args.completed=len(results)
                status(args,phase='group_complete',group=result['group'])
                summarize(args,results)
    save(args.output_dir/'complete.json',dict(new_training_groups=GROUPS,reused_reference_groups=len(args.references),
         epochs_per_group=EPOCHS,final_test_access=False,elapsed_minutes=(time.time()-args.started)/60))
    status(args,phase='complete')
    print(f'PSEUDO_COMPLETE groups={GROUPS}/{GROUPS} elapsed_hours={(time.time()-args.started)/3600:.2f}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    defaults=dict(private_dir='data/new_data/merged',nyuv2_dir='data/public_semseg/nyuv2/processed',
        private_source='data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B1',
        public_source='data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2',
        task_dir='data/paper_benchmark_runs/task_control_window_v1',
        output_dir='data/paper_benchmark_runs/pseudo_intensity_v1')
    for key,value in defaults.items():
        p.add_argument('--'+key.replace('_','-'),type=Path,default=Path(value))
    p.add_argument('--device',default='cuda')
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    lock=args.output_dir/'RUNNING.lock'
    with lock.open('x',encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(),host=socket.gethostname(),started=time.time()),handle)
    try:
        execute(args)
    except BaseException:
        error=traceback.format_exc()
        save(args.output_dir/'failure.json',dict(error=error))
        print('PSEUDO_FAILED '+error,flush=True)
        raise
    finally:
        lock.replace(args.output_dir/f'RUNNING.finished.{time.time_ns()}.json')


if __name__=='__main__':
    main()
