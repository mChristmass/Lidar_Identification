"""Budgeted paired CMX regional-channel / semantic exchange experiment."""
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
from .model import CMXPaper
from .data import read_manifest
from .run import seed_all, save_json
from .run_structure_window import PrivateTrain, load_source, source_path
from .train_mixed_control import TrainingData, atomic_checkpoint
from .diagnose_phase1 import digest
from .regional_exchange import attach, probe_loss, evidence
from .regional_exchange_diagnostics import snapshot


def build(args, domain, seed, arm):
    source = load_source(args, domain, seed)
    seed_all(seed)
    model = CMXPaper(source['args']['model'], 2 if domain=='private' else 40,
                     128 if domain=='private' else 640, source['args']['decoder_dim'])
    model.load_state_dict(source['model'], strict=True)
    # Same probes initialize identically for every arm, independent of source loading RNG.
    seed_all(seed+10000)
    return attach(model, 2 if domain=='private' else 40, arm).to(args.device)


def dataset(args, domain, seed):
    return PrivateTrain(args.private_dir, seed) if domain=='private' else TrainingData(args.nyuv2_dir, 'mixed', seed)


def loader(data, domain, seed, epoch):
    data.epoch = epoch
    return DataLoader(data, batch_size=8 if domain=='private' else 2, num_workers=0,
                      pin_memory=False, shuffle=True, generator=torch.Generator().manual_seed(seed+epoch))


def expired(args):
    return time.monotonic() >= args.deadline


def summarize(rows):
    paired=[]
    for domain in ('nyuv2','private'):
        base={r['seed']:r for r in rows if r['domain']==domain and r['arm']=='B'}
        for arm in ('P','G','H','PG'):
            candidates=[r for r in rows if r['domain']==domain and r['arm']==arm and r['seed'] in base]
            if not candidates: continue
            for condition in candidates[0]['conditions']:
                delta=[100*(r['conditions'][condition]['miou']-base[r['seed']]['conditions'][condition]['miou']) for r in candidates]
                paired.append(dict(domain=domain,arm=arm,condition=condition,seeds=[r['seed'] for r in candidates],
                    delta_pp=delta,mean_delta_pp=float(np.mean(delta)),
                    std_delta_pp=float(np.std(delta,ddof=1)) if len(delta)>1 else None))
    return paired


def calibration(args, domain, seed, protocol):
    path = args.output_dir/'calibration'/f'{domain}_{seed}.pth'
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['protocol'] != protocol: raise ValueError('Calibration protocol changed')
        return state
    print(f'EXCHANGE_CALIBRATE domain={domain} seed={seed}', flush=True)
    model = build(args, domain, seed, 'B')
    model.requires_grad_(False)
    heads = [p for m in model.backbone.FRMs for p in m.probes.parameters()]
    for p in heads: p.requires_grad_(True)
    optimizer = torch.optim.AdamW(heads, lr=1e-3)
    data = dataset(args, domain, seed)
    for epoch in range(args.probe_epochs):
        model.eval(); seed_all(seed+epoch)
        for x,y in loader(data, domain, seed, epoch):
            if expired(args): return None
            model(x.to(args.device)); loss = probe_loss(model, y.to(args.device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    scores = [[] for _ in range(4)]
    model.eval()
    with torch.no_grad():
        for index,(x,y) in enumerate(loader(data, domain, seed, 0)):
            if index >= 24: break
            model(x.to(args.device))
            for stage,m in enumerate(model.backbone.FRMs):
                c,_,_ = evidence(*m.logits)
                labels = F.interpolate(y[:,None].float().to(args.device), c.shape[-2:], mode='nearest').long()
                # Equal per-class cap prevents background-only threshold calibration.
                for cls in labels.unique():
                    if int(cls)==255: continue
                    values = c[labels==cls].flatten()
                    if values.numel():
                        take = torch.linspace(0, values.numel()-1, min(64,values.numel()),device=values.device).long()
                        scores[stage].extend(values[take].cpu().tolist())
    state = dict(protocol=protocol,
                 probes=[{k:v.cpu() for k,v in m.probes.state_dict().items()} for m in model.backbone.FRMs],
                 thresholds=[float(np.quantile(s,.75)) for s in scores])
    path.parent.mkdir(parents=True, exist_ok=True); atomic_checkpoint(path, state)
    save_json(path.with_suffix('.json'), dict(thresholds=state['thresholds'],
        probe_epochs=args.probe_epochs, calibration='training only, per-class capped samples, 75th percentile',
        interpretation='Frozen detached semantic probes; entropy confidence is not calibrated correctness.'))
    del model, optimizer
    return state


def train(args, domain, seed, arm, protocol):
    folder = args.output_dir/domain/f'{arm}_seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
    result_path, last = folder/'result.json', folder/'last.pth'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['checkpoint_sha256'] != digest(last): raise ValueError('Completed checkpoint changed')
        return result
    cal = calibration(args, domain, seed, protocol)
    if cal is None or expired(args): return None
    model = build(args, domain, seed, arm)
    for m,heads,threshold in zip(model.backbone.FRMs,cal['probes'],cal['thresholds']):
        m.probes.load_state_dict(heads); m.probes.requires_grad_(False); m.threshold.fill_(threshold)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=.01)
    amp = args.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda',enabled=amp)
    first,history = 1,[]
    if last.exists():
        state = torch.load(last,map_location='cpu',weights_only=False)
        if state['protocol'] != protocol: raise ValueError('Resume protocol changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler']); first=state['epoch']+1; history=state['history']
    elif not (folder/'diagnostics/initial/statistics.json').exists():
        snapshot(args,model,domain,folder,'initial')
    data=dataset(args,domain,seed)
    for epoch in range(first,args.epochs+1):
        if expired(args): return None
        seed_all(seed+epoch); model.train()
        for m in model.backbone.FRMs: m.probes.eval()
        batches=loader(data,domain,seed,epoch); total=0.; started=time.monotonic()
        accum=1 if domain=='private' else 4; batch=8 if domain=='private' else 2
        optimizer.zero_grad(set_to_none=True)
        for index,(x,y) in enumerate(batches):
            # Epoch checkpoints allow exact replay from deterministic epoch seeds.
            if expired(args): return None
            x,y=x.to(args.device),y.to(args.device)
            for group in optimizer.param_groups:
                group['lr']=args.lr*(1-((epoch-1)+index/len(batches))/args.epochs)**.9
            window=min(batch*accum,len(data)-(index//accum)*batch*accum)
            with torch.autocast(device_type='cuda',enabled=amp):
                loss=F.cross_entropy(model(x)['logits'],y.long(),ignore_index=255)
            if not torch.isfinite(loss): raise FloatingPointError(f'{domain}/{arm}/{seed}')
            scaler.scale(loss*len(x)/window).backward()
            if (index+1)%accum==0 or index+1==len(batches):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for m in model.backbone.FRMs: m.alpha.clamp_(0,1)
            total+=float(loss.detach())*len(x)
        history.append(dict(epoch=epoch,loss=total/len(data),seconds=time.monotonic()-started))
        atomic_checkpoint(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),epoch=epoch,history=history,protocol=protocol))
        save_json(folder/'history.json',history)
        save_json(args.output_dir/'status.json',dict(phase='training',domain=domain,arm=arm,seed=seed,
                  epoch=epoch,epochs=args.epochs,updated_unix=time.time()))
        print(f'EXCHANGE_PROGRESS {domain}/{arm}/seed{seed} epoch={epoch}/{args.epochs} loss={total/len(data):.5f}',flush=True)
        if epoch==args.epochs//2:
            atomic_checkpoint(folder/'midpoint.pth',dict(model=model.state_dict(),epoch=epoch,protocol=protocol))
            snapshot(args,model,domain,folder,'midpoint')
    if expired(args): return None
    save_json(args.output_dir/'status.json',dict(phase='evaluating',domain=domain,arm=arm,seed=seed,updated_unix=time.time()))
    conditions=snapshot(args,model,domain,folder,'final',full=True)
    result=dict(domain=domain,arm=arm,seed=seed,conditions=conditions,checkpoint_sha256=digest(last))
    save_json(result_path,result)
    print('EXCHANGE_GROUP_COMPLETE '+json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-dir',type=Path,default=Path('data/new_data/merged'))
    p.add_argument('--nyuv2-dir',type=Path,default=Path('data/public_semseg/nyuv2/processed'))
    p.add_argument('--private-source',type=Path,default=Path('data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B1'))
    p.add_argument('--public-source',type=Path,default=Path('data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2'))
    p.add_argument('--output-dir',type=Path,default=Path('data/paper_benchmark_runs/regional_exchange_v1'))
    p.add_argument('--budget-hours',type=float,default=12)
    p.add_argument('--device',default='cuda'); p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--probe-epochs',type=int,default=2); p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--visual-samples',type=int,default=12)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    if min(args.epochs,args.probe_epochs,args.visual_samples,args.budget_hours,args.lr)<=0: raise ValueError('Positive arguments required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    args.deadline=time.monotonic()+args.budget_hours*3600
    queue=[(domain,seed,arm) for seed in (42,777,2025)
           for arm in ('B','G','H','P','PG') for domain in ('nyuv2','private')]
    files=[Path(__file__),Path(__file__).with_name('regional_exchange.py'),
           Path(__file__).with_name('regional_exchange_diagnostics.py')]
    files.extend(Path(__file__).with_name(n) for n in ('model.py','data.py','run.py','train_mixed_control.py',
        'run_structure_window.py','corruptions_v2.py','diagnose_phase1.py'))
    files.extend([Path(__file__).parents[1]/'cmx_initial_transfer/official'/n for n in ('net_utils.py','dual_segformer.py','mlp_decoder.py')])
    for seed in (42,777,2025):
        for domain in ('nyuv2','private'): files.append(source_path(args,domain,seed))
    split=args.private_dir/'paper_split_v1'
    train_ids,dev_ids=[np.load(split/f'{s}_indices.npy') for s in ('train','dev')]
    if len(train_ids)!=970 or len(dev_ids)!=137 or np.intersect1d(train_ids,dev_ids).size: raise ValueError('Private split mismatch')
    files.extend([split/'train_indices.npy',split/'dev_indices.npy'])
    files.append(split/'frame_acquisition_metadata.csv')
    files.extend([args.private_dir/'intensity.npy',args.private_dir/'depth.npy'])
    files.extend(args.private_dir/'label'/f'{int(i)}.png' for i in np.concatenate((train_ids,dev_ids)))
    TrainingData(args.nyuv2_dir,'mixed',42)
    files.extend([args.nyuv2_dir/'manifests'/f'{s}.jsonl' for s in ('train','dev')])
    protocol=dict(version=1,epochs=args.epochs,lr=args.lr,probe_epochs=args.probe_epochs,
        visual_samples=args.visual_samples,queue=queue,hashes={str(f):digest(f) for f in files},
        selection='fixed final epoch, train/dev only',public_train='50% clean + random corruption',
        probe='shared train-only calibration then frozen; features detached',
        gate='G: r_source*(c+(1-c)*(1-r_target)); H: c>=train_q75; c=1-JS/log2',
        local_ca='4x4 pooled regions, original MLP shared, bilinear weights, projected alpha in [0,1]')
    # Canonicalize tuples before comparing JSON on resume.
    protocol=json.loads(json.dumps(protocol))
    path=args.output_dir/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError('Protocol changed; choose a new output directory')
    save_json(path,protocol)
    if args.preflight_only:
        print('EXCHANGE_PREFLIGHT_OK groups=30',flush=True); return
    lock=args.output_dir/'RUNNING.lock'
    with lock.open('x') as f: json.dump(dict(pid=os.getpid(),host=socket.gethostname()),f)
    rows=[]
    try:
        for domain,seed,arm in queue:
            if expired(args): break
            row=train(args,domain,seed,arm,protocol)
            if row is None: break
            rows.append(row)
            save_json(args.output_dir/'comparison.json',dict(groups=rows,paired=summarize(rows),completed=len(rows),total=len(queue)))
            print(f'EXCHANGE_OVERALL completed={len(rows)}/{len(queue)}',flush=True)
            if args.device.startswith('cuda'): torch.cuda.empty_cache()
        phase='complete' if len(rows)==len(queue) else 'budget_paused'
        save_json(args.output_dir/'status.json',dict(phase=phase,completed=len(rows),total=len(queue),updated_unix=time.time()))
        print(f'EXCHANGE_STOP phase={phase} completed={len(rows)}/{len(queue)}',flush=True)
    except Exception:
        save_json(args.output_dir/'failure.json',dict(error=traceback.format_exc(),updated_unix=time.time()))
        save_json(args.output_dir/'status.json',dict(phase='failed',completed=len(rows),updated_unix=time.time()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__': main()
