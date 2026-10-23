"""Paired per-frame private foreground IoU and labeled oracle diagnostics."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ARMS = ('P', 'G', 'H', 'PG')
SEEDS = (42, 777, 2025)


def read_masks(root, arm, seed):
    path = root / 'private' / f'{arm}_seed{seed}' / 'diagnostics/final/natural_predictions.npz'
    with np.load(path) as z:
        return z['sample_ids'].astype(int), z['pred'].astype(bool)


def counts(gt, pred):
    return dict(tp=int((gt & pred).sum()), fp=int((~gt & pred).sum()),
                fn=int((gt & ~pred).sum()), tn=int((~gt & ~pred).sum()))


def fg_iou(c):
    union = c['tp'] + c['fp'] + c['fn']
    return c['tp'] / union if union else None


def aggregate(matrix):
    tn, fp, fn, tp = (int(x) for x in matrix)
    bg = tn / max(tn + fp + fn, 1)
    fg = tp / max(tp + fp + fn, 1)
    return dict(miou=(bg + fg)/2, foreground_iou=fg,
                confusion_matrix=[[tn, fp], [fn, tp]])


def panel(path, frame, gt, base, candidate, intensity, depth, title):
    scale = 3
    arrays = [intensity, depth, gt, base, candidate,
              np.where((~base & candidate) & gt, 1,
                       np.where((base & ~candidate) & gt, 2,
                                np.where((base & ~candidate) & ~gt, 3,
                                         np.where((~base & candidate) & ~gt, 4, 0))))]
    labels = ['Intensity', 'Depth', 'GT', 'CMX B', title, '1 rescue 2 lost 3 FP fixed 4 new FP']
    canvas = Image.new('RGB', (128*scale*3, (128*scale+26)*2), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (name, array) in enumerate(zip(labels, arrays)):
        if index < 2:
            lo, hi = np.nanpercentile(array, [1, 99])
            v = (np.nan_to_num(array)-lo)/max(hi-lo,1e-6)
            rgb = np.repeat((np.clip(v,0,1)*255).astype(np.uint8)[...,None],3,-1)
        elif index == 5:
            colors=np.asarray([[0,0,0],[0,210,0],[240,160,0],[0,190,255],[255,0,0]],np.uint8)
            rgb=colors[array.astype(int)]
        else:
            rgb=np.repeat((array.astype(np.uint8)*255)[...,None],3,-1)
        x,y=(index%3)*128*scale,(index//3)*(128*scale+26)
        draw.text((x+3,y+4),name,fill='black')
        canvas.paste(Image.fromarray(rgb).resize((128*scale,128*scale),Image.Resampling.NEAREST),(x,y+26))
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-dir',type=Path,default=Path('data/paper_benchmark_runs/regional_exchange_v1'))
    p.add_argument('--private-dir',type=Path,default=Path('data/new_data/merged'))
    p.add_argument('--output-dir',type=Path,default=Path('data/analysis/regional_exchange_frame_audit_v1'))
    args=p.parse_args()
    if not (args.runs_dir/'status.json').is_file() or json.loads((args.runs_dir/'status.json').read_text())['phase']!='complete':
        raise ValueError('The regional-exchange run is not complete')
    ids=np.load(args.private_dir/'paper_split_v1/dev_indices.npy').astype(int)
    masks={}
    for seed in SEEDS:
        for arm in ('B',*ARMS):
            got,pred=read_masks(args.runs_dir,arm,seed)
            if not np.array_equal(got,ids): raise ValueError(f'Frozen frame IDs differ: {arm}/{seed}')
            masks[(seed,arm)]=pred
    gt={int(i):np.asarray(Image.open(args.private_dir/'label'/f'{i}.png'))==1 for i in ids}
    metadata={int(r['frame_1based']):r for r in csv.DictReader(
        (args.private_dir/'paper_split_v1/frame_acquisition_metadata.csv').open(encoding='utf-8-sig'))}
    rows=[]; summaries=[]
    for seed in SEEDS:
        for arm in ARMS:
            base_total=np.zeros(4,np.int64); arm_total=np.zeros(4,np.int64); oracle_total=np.zeros(4,np.int64)
            positive=[]; negative=[]; ties=empty=empty_better=empty_worse=0
            for index,frame in enumerate(ids):
                target=gt[int(frame)]; b=masks[(seed,'B')][index]; a=masks[(seed,arm)][index]
                bc,ac=counts(target,b),counts(target,a)
                base_total += [bc[k] for k in ('tn','fp','fn','tp')]
                arm_total += [ac[k] for k in ('tn','fp','fn','tp')]
                bi,ai=fg_iou(bc),fg_iou(ac)
                if target.any():
                    delta=ai-bi
                    selected=ac if delta>1e-12 else bc
                    if delta>1e-12: positive.append(delta)
                    elif delta< -1e-12: negative.append(delta)
                    else: ties+=1
                else:
                    empty+=1; delta=None
                    if ac['fp']<bc['fp']: empty_better+=1
                    if ac['fp']>bc['fp']: empty_worse+=1
                    selected=ac if ac['fp']<bc['fp'] else bc
                oracle_total += [selected[k] for k in ('tn','fp','fn','tp')]
                area=int(target.sum())
                size='empty' if not area else '<=80' if area<=80 else '81-160' if area<=160 else '161-320' if area<=320 else '>320'
                rows.append(dict(seed=seed,arm=arm,frame_id=int(frame),primary_class=metadata[int(frame)]['primary_class'],
                    size_bin=size,foreground_pixels=area,
                    baseline_iou=bi,arm_iou=ai,delta_iou=delta,baseline_fp=bc['fp'],arm_fp=ac['fp'],
                    rescued_tp=int((target & ~b & a).sum()),lost_tp=int((target & b & ~a).sum()),
                    fixed_fp=int((~target & b & ~a).sum()),new_fp=int((~target & ~b & a).sum())))
            summaries.append(dict(seed=seed,arm=arm,positive_frames=len(positive),negative_frames=len(negative),
                tied_frames=ties,foreground_frames=len(positive)+len(negative)+ties,empty_frames=empty,
                empty_fp_improved=empty_better,empty_fp_worsened=empty_worse,
                positive_mean_delta=float(np.mean(positive)) if positive else None,
                positive_median_delta=float(np.median(positive)) if positive else None,
                positive_max_delta=float(max(positive)) if positive else None,
                negative_mean_delta=float(np.mean(negative)) if negative else None,
                baseline=aggregate(base_total),arm_metrics=aggregate(arm_total),
                label_oracle=aggregate(oracle_total)))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'paired_frames.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    consensus=[]
    for arm in ARMS:
        for frame in ids:
            matched=[r for r in rows if r['arm']==arm and r['frame_id']==frame]
            if matched[0]['delta_iou'] is None: continue
            delta=[r['delta_iou'] for r in matched]
            consensus.append(dict(arm=arm,frame_id=int(frame),mean_delta=float(np.mean(delta)),
                improved_seeds=sum(x>1e-12 for x in delta),worsened_seeds=sum(x< -1e-12 for x in delta)))
    report=dict(protocol='Frozen private dev, 3 paired seeds, foreground-containing frames only for per-frame IoU; empty frames tracked by FP count',
        summaries=summaries,consensus=consensus,oracle_note='Per-frame selection uses GT and is an unattainable upper bound, not deployable performance.')
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    # Global extremes across all paired seeds, plus seed-42 examples.
    intensity=np.load(args.private_dir/'intensity.npy',mmap_mode='r')
    depth=np.load(args.private_dir/'depth.npy',mmap_mode='r')
    for arm in ARMS:
        candidates=[r for r in rows if r['arm']==arm and r['delta_iou'] is not None]
        seed42=[r for r in candidates if r['seed']==42]
        choices=[('global_best',max(candidates,key=lambda r:r['delta_iou'])),
                 ('global_worst',min(candidates,key=lambda r:r['delta_iou'])),
                 ('seed42_best',max(seed42,key=lambda r:r['delta_iou'])),
                 ('seed42_worst',min(seed42,key=lambda r:r['delta_iou']))]
        for kind,row in choices:
            frame=row['frame_id']; idx=int(np.where(ids==frame)[0][0])
            seed=row['seed']
            panel(args.output_dir/'figures'/f'{arm}_{kind}_frame{frame}_seed{seed}.png',frame,gt[frame],
                  masks[(seed,'B')][idx],masks[(seed,arm)][idx],intensity[frame-1],depth[frame-1],
                  f'{arm} IoU {row["arm_iou"]:.3f} vs B {row["baseline_iou"]:.3f}')
    print('FRAME_AUDIT_COMPLETE',args.output_dir,flush=True)


if __name__=='__main__': main()
