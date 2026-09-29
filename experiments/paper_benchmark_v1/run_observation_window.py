"""Joint private/public exploratory matrix, eight-hour soft budget, no final test."""
import argparse
import json
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .corruptions import stable_seed
from .data import PrivatePaperDataset, read_manifest
from .diagnose_phase1 import DiagnosticData, digest, regional
from .observation_model import ObservationCMX, observation_descriptors, DESCRIPTORS
from .run import seed_all, evaluate
from .run_noise_window import CONDITIONS
from .train_mixed_control import TrainingData, atomic_checkpoint
from .train_reliability import read, save

METHODS = ('B0', 'L', 'E', 'C')
EPOCHS = 20


class PrivateTrain(PrivatePaperDataset):
    def __init__(self, root, seed):
        super().__init__(root, root/'paper_split_v1/train_indices.npy', training=False)
        self.seed, self.epoch = seed, 0
    def __getitem__(self, index):
        item = super().__getitem__(index)
        rng = np.random.default_rng(stable_seed(self.seed, 'train', item['sample_id'], 'flip', str(self.epoch)))
        x, y = item['inputs'], item['labels']
        if rng.random()<.5:
            x, y = x.flip(-1), y.flip(-1)
        return x, y


def update(args, **state):
    save(args.output_dir/'status.json', dict(updated_unix=time.time(),
         groups_completed=getattr(args, 'completed', 0), groups_possible=16, **state))


def source_path(args, dataset, seed):
    return (args.private_source if dataset=='private' else args.public_source)/f'seed_{seed}/best.pth'


def load_source(args, dataset, seed):
    source = torch.load(source_path(args, dataset, seed), map_location='cpu', weights_only=False)
    expected = dict(dataset=dataset, model='CMX_B1' if dataset=='private' else 'CMX_B2',
                    stage='tune', seed=seed, height=128 if dataset=='private' else 480,
                    width=128 if dataset=='private' else 640)
    if dataset=='nyuv2':
        expected['corruption'] = 'clean'
    if any(source['args'].get(k)!=v for k, v in expected.items()):
        raise ValueError(f'Wrong source: {dataset}/{seed}')
    return dict(model=source['model'], args=source['args'])


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
        for seed in (42, 777):
            files[f'source/{dataset}/{seed}'] = source_path(args, dataset, seed)
            source = load_source(args, dataset, seed)
            del source
    fingerprints = {key: digest(path) for key, path in files.items()}
    for key, value in prior['private_sha256'].items():
        if key!='checkpoint' and fingerprints['private/'+key]!=value:
            raise ValueError(f'Private frozen data changed: {key}')
    package = Path(__file__).parent
    code = sorted(package.glob('*.py'))+sorted((package.parent/'cmx_initial_transfer/official').glob('*.py'))
    protocol = dict(version=1, fingerprints=fingerprints,
                    code={str(p.relative_to(package.parent)): digest(p) for p in code},
                    epochs=EPOCHS, methods=list(METHODS), seeds=[42,777], budget_hours=args.budget_hours,
                    scope='dev-only broad screen; BOTH datasets participate; no efficacy gate',
                    sources='private best B1 / public clean best B2, identical per dataset/seed across methods',
                    training='private natural deterministic flip; public mixed v2, CE only; fixed final20',
                    optimizer='AdamW base1e-5 added1e-3 wd.01 poly.9 clip1; batch8 private / batch2 accum4 public',
                    descriptors=list(DESCRIPTORS), interpretation='normalized structural descriptors, not posterior uncertainty',
                    second_seed_rule='all methods both domains; admit only if remaining >=1.25*seed42 walltime+10min')
    path = args.output_dir/'protocol.json'
    if path.exists() and read(path)!=protocol:
        raise ValueError('Protocol/input/code changed: use a new output directory')
    save(path, protocol)
    TrainingData(args.nyuv2_dir, 'mixed', 42)
    return protocol


def audit_inputs(args):
    """Deterministically selected train frames; descriptive QA, not reliability validation."""
    destination = args.output_dir/'input_audit'
    for dataset in ('private', 'nyuv2'):
        data = (PrivateTrain(args.private_dir, 42) if dataset=='private' else
                TrainingData(args.nyuv2_dir, 'clean', 42))
        rows = []
        for index in np.linspace(0, len(data)-1, 6, dtype=int):
            x, y = data[int(index)]
            desc = observation_descriptors(x[None])[0]
            if not torch.isfinite(desc).all():
                raise ValueError('Non-finite observation descriptors')
            panels = [x[0], x[1], x[2], *desc]
            labels = ['intensity', 'depth', 'valid', *DESCRIPTORS]
            canvas = Image.new('RGB', (160*len(panels), 180), 'white')
            draw = ImageDraw.Draw(canvas)
            for n, (panel, label) in enumerate(zip(panels, labels)):
                pic = Image.fromarray((panel.numpy().clip(0,1)*255).astype(np.uint8)).resize((160,160))
                canvas.paste(pic, (n*160,20))
                draw.text((n*160,2), label, fill='black')
            destination.mkdir(parents=True, exist_ok=True)
            canvas.save(destination/f'{dataset}_train_{index}.png')
            rows.append(dict(train_index=int(index), component_means=desc.mean((1,2)).tolist()))
        save(destination/f'{dataset}.json', dict(names=list(DESCRIPTORS), rows=rows,
             note='Shared fixed0..1 display; not per-image rescaling; maps are descriptive, not calibrated'))


def component_counts(pred, truth):
    """8-connected GT components; IoU>=.5 one-to-one matching, NOT object identities."""
    gt, count = ndimage.label(truth, np.ones((3,3)))
    pr, _ = ndimage.label(pred, np.ones((3,3)))
    gt_area, pr_area = np.bincount(gt.ravel()), np.bincount(pr.ravel())
    candidates = []
    for g in range(1, count+1):
        ids, intersections = np.unique(pr[gt==g], return_counts=True)
        for p, intersection in zip(ids, intersections):
            if p:
                iou = intersection/(gt_area[g]+pr_area[p]-intersection)
                if iou>=.5:
                    candidates.append((iou, g, p))
    matched_gt, matched_pr = set(), set()
    for _, g, p in sorted(candidates, reverse=True):
        if g not in matched_gt and p not in matched_pr:
            matched_gt.add(g); matched_pr.add(p)
    tiny = {g for g in range(1,count+1) if gt_area[g]<=80}
    return dict(components=count, detected=len(matched_gt), tiny_components=len(tiny),
                tiny_detected=len(tiny & matched_gt))


@torch.no_grad()
def private_details(model, data, device):
    rows = []
    for item in data:
        pred = model(item['inputs'][None].to(device))['logits'].argmax(1)[0].cpu().numpy()==1
        truth = item['labels'].numpy()==1
        tp, fp, fn = int((pred&truth).sum()), int((pred&~truth).sum()), int((~pred&truth).sum())
        rows.append(dict(sample_id=item['sample_id'], tp=tp, fp=fp, fn=fn,
                         **component_counts(pred, truth)))
    totals = {k: sum(r[k] for r in rows) for k in ('components','detected','tiny_components','tiny_detected')}
    totals['component_recall_iou50'] = totals['detected']/totals['components'] if totals['components'] else None
    totals['tiny_component_recall_iou50_area80'] = totals['tiny_detected']/totals['tiny_components'] if totals['tiny_components'] else None
    return dict(rows=rows, **totals)


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
    model = ObservationCMX(method, source['args']['model'], 2 if dataset=='private' else 40,
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
        for index, (x,y) in enumerate(loader):
            x, y = x.to(args.device), y.to(args.device)
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr']*(1-((epoch-1)+index/len(loader))/EPOCHS)**.9
            window_samples = min(batch*accum, len(data)-(index//accum)*batch*accum)
            with torch.autocast(device_type='cuda', enabled=amp):
                ce = F.cross_entropy(model(x)['logits'], y, ignore_index=255)
                loss = ce*len(x)/window_samples
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{name}/{epoch}/{index}')
            scaler.scale(loss).backward()
            if (index+1)%accum==0 or index+1==len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            loss_sum += float(ce.detach())*len(x)
            if index%25==0 or index+1==len(loader):
                update(args, phase='training', group=name, epoch=epoch, epochs=EPOCHS,
                       batch=index+1, batches=len(loader), loss=loss_sum/min((index+1)*batch,len(data)))
        history.append(dict(epoch=epoch, loss=loss_sum/len(data)))
        atomic_checkpoint(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scaler=scaler.state_dict(), history=history, epoch=epoch, protocol=protocol, group=name))
        save(destination/'history.json', dict(epochs=history))
        print(f'OBS_EPOCH {name} epoch={epoch}/{EPOCHS} loss={history[-1]["loss"]:.6f}', flush=True)
    model.eval()
    fingerprint = digest(last)
    result = dict(dataset=dataset, method=method, seed=seed, group=name,
                  checkpoint_sha256=fingerprint, added_parameters=sum(p.numel() for p in added))
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
    result['elapsed_minutes'] = (time.time()-started)/60
    save(result_path,result)
    print(f'OBS_GROUP_COMPLETE {name} result={json.dumps(result.get("metrics", {"clean":result.get("clean_miou"),"corrupt_mean":result.get("corruption_mean_miou")}))}',flush=True)
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
            print('OBS_SUMMARY '+json.dumps(summary),flush=True)
    save(args.output_dir/'comparison.json',dict(groups=results,summaries=summaries,
         interpretation='Separate domain endpoints, no pooled score or success threshold; FP lower is better; seed spread is not scene generalization'))


def execute(args):
    started = time.time()
    update(args,phase='preflight')
    protocol = preflight(args)
    audit_inputs(args)
    results = []
    seed42_started = time.time()
    planned_seeds = [42]
    for seed in (42,777):
        if seed==777:
            path = args.output_dir/'second_seed_decision.json'
            if path.exists():
                decision = read(path)
            else:
                elapsed = max((time.time()-seed42_started)/60,
                              sum(r['elapsed_minutes'] for r in results if r['seed']==42))
                remaining = args.budget_hours*60-(time.time()-started)/60
                decision = dict(run=remaining>=1.25*elapsed+10, first_seed_minutes=elapsed,
                                remaining_minutes=remaining, basis='runtime only, not scores')
                save(path,decision)
            if not decision['run']:
                break
            planned_seeds.append(777)
        for method in METHODS:
            for dataset in ('nyuv2','private'):
                args.completed=len(results)
                results.append(train_group(args,dataset,method,seed,protocol))
                summarize(args,results)
    save(args.output_dir/'complete.json',dict(training_groups=len(results), planned_seeds=planned_seeds,
         public_condition_evaluations=22*sum(r['dataset']=='nyuv2' for r in results),
         private_evaluations=sum(r['dataset']=='private' for r in results),
         elapsed_minutes=(time.time()-started)/60, final_test_access=False))
    args.completed=len(results)
    update(args,phase='complete',groups_planned=len(results),seeds=planned_seeds)
    print(f'OBS_COMPLETE groups={len(results)} seeds={planned_seeds}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    defaults=dict(private_dir='data/new_data/merged',nyuv2_dir='data/public_semseg/nyuv2/processed',
        private_source='data/paper_benchmark_runs/phase1_v1/private/tune/CMX_B1',
        public_source='data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2',
        repair_dir='data/paper_benchmark_runs/repair_window_v1',output_dir='data/paper_benchmark_runs/observation_window_v1')
    for k,v in defaults.items():
        parser.add_argument('--'+k.replace('_','-'),type=Path,default=Path(v))
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--budget-hours',type=float,default=8.)
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
