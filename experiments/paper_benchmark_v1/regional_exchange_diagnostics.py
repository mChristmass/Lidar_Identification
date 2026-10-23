"""Numeric snapshots plus fixed-scale PNG contact sheets; no test split access."""
import json
import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from .data import PrivatePaperDataset
from .diagnose_phase1 import DiagnosticData
from .run import evaluate, save_json
from torch.utils.data import DataLoader


def datasets(args, domain):
    if domain == 'private':
        return {'natural': PrivatePaperDataset(args.private_dir, args.private_dir/'paper_split_v1/dev_indices.npy')}
    return {f'{f}_{s}': DiagnosticData(args.nyuv2_dir, f, s)
            for f, s in [('clean', 'medium'), ('joint', 'light'), ('joint', 'medium'),
                         ('joint', 'heavy'), ('photon_proxy', 'heavy'), ('block_missing', 'heavy')]}


def color(a, continuous=False):
    if continuous == 'gray':
        return np.repeat(np.clip(a,0,1)[...,None],3,-1)*255
    if continuous == 'correction':
        rgb=np.zeros((*a.shape,3)); rgb[a==1]=(0,220,0); rgb[a==2]=(255,0,0)
        return rgb
    if continuous:
        v = np.clip(a, 0, 1)
        return np.stack((v, 1-np.abs(2*v-1), 1-v), -1)*255
    a = a.astype(np.int64)
    rgb = np.stack(((a*67)%255, (a*131)%255, (a*197)%255), -1)
    rgb[a == 255] = 127
    return rgb


def sheet(path, panels):
    w, h = 240, 200
    canvas = Image.new('RGB', (w*4, 24+(h+24)*((len(panels)+3)//4)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((3,4),'Heatmaps: blue=0 green=0.5 red=1. Correction: green=rescue red=damage. CA channel IDs fixed.',fill='black')
    for i, (title, a, continuous) in enumerate(panels):
        rgb = color(a, continuous).astype(np.uint8)
        pic = Image.fromarray(rgb).resize((w, h), Image.Resampling.NEAREST)
        x, y = (i%4)*w, 24+(i//4)*(h+24)
        canvas.paste(pic, (x, y+24)); draw.text((x+3, y+4), title, fill='black')
    canvas.save(path)


@torch.no_grad()
def snapshot(args, model, domain, folder, phase, full=False):
    model.eval()
    root = folder/'diagnostics'/phase
    root.mkdir(parents=True, exist_ok=True)
    records, conditions = [], {}
    classes = 2 if domain == 'private' else 40
    for condition, data in datasets(args, domain).items():
        all_masks, all_ids = [], []
        if full:
            conditions[condition] = evaluate(model, DataLoader(data, batch_size=1, num_workers=0), args.device, classes)
        # Fixed IDs chosen before scores: evenly spaced, shared across arms/seeds/conditions.
        candidates = []
        if domain == 'private':
            meta = {int(r['frame_1based']):r for r in csv.DictReader(
                (args.private_dir/'paper_split_v1/frame_acquisition_metadata.csv').open(encoding='utf-8-sig'))}
            seen = set()
            for i, frame in enumerate(data.ids):
                gt = np.asarray(Image.open(args.private_dir/'label'/f'{frame}.png')) == 1
                area = int(gt.sum())
                size = 'empty' if area==0 else ('<=80' if area<=80 else '81-160' if area<=160 else '161-320' if area<=320 else '>320')
                keys = ['class/'+meta[int(frame)]['primary_class'], 'size/'+size]
                if any(k not in seen for k in keys): candidates.append(i)
                seen.update(keys)
        candidates.extend(np.linspace(0, len(data)-1, min(args.visual_samples, len(data)), dtype=int).tolist())
        chosen = set(list(dict.fromkeys(candidates))[:args.visual_samples])
        indices = range(len(data)) if full else sorted(chosen)
        for idx in indices:
            item = data[idx]; x = item['inputs'][None].to(args.device)
            y = item['labels'].numpy(); ok = y != 255
            for m in model.backbone.FRMs: m.capture = True
            pred = model(x)['logits'].argmax(1)[0].cpu().numpy()
            sample = str(item['sample_id'])
            if full:
                all_masks.append(pred.astype(np.uint8)); all_ids.append(sample)
            arrays = dict(inputs=item['inputs'].numpy(), gt=y, pred=pred)
            for key in ('original_valid','added'):
                if key in item: arrays[key] = item[key].numpy()
            panels = [('Intensity', arrays['inputs'][0], 'gray'), ('Depth', arrays['inputs'][1], True),
                      ('GT', y, False), ('Fusion', pred, False)]
            for stage, m in enumerate(model.backbone.FRMs):
                ps = [F.interpolate(z.float(), y.shape, mode='bilinear', align_corners=False).softmax(1)[0] for z in m.logits]
                pi, pd = [z.argmax(0).cpu().numpy() for z in ps]
                quadrant = (pi == y).astype(np.uint8)*2 + (pd == y).astype(np.uint8)
                row = dict(sample_id=sample, condition=condition, stage=stage,
                           probe_i_correct=int(((pi==y)&ok).sum()), probe_d_correct=int(((pd==y)&ok).sum()),
                           valid_pixels=int(ok.sum()))
                row['fusion_confusion'] = np.bincount(y[ok]*classes+pred[ok],minlength=classes*classes).reshape(classes,classes).tolist()
                row['probe_i_confusion'] = np.bincount(y[ok]*classes+pi[ok],minlength=classes*classes).reshape(classes,classes).tolist()
                row['probe_d_confusion'] = np.bincount(y[ok]*classes+pd[ok],minlength=classes*classes).reshape(classes,classes).tolist()
                if stage==0 and domain=='nyuv2':
                    boundary=np.zeros(y.shape,bool)
                    boundary[1:] |= (y[1:]!=y[:-1]) & ok[1:] & ok[:-1]
                    boundary[:-1] |= (y[1:]!=y[:-1]) & ok[1:] & ok[:-1]
                    boundary[:,1:] |= (y[:,1:]!=y[:,:-1]) & ok[:,1:] & ok[:,:-1]
                    boundary[:,:-1] |= (y[:,1:]!=y[:,:-1]) & ok[:,1:] & ok[:,:-1]
                    regions=dict(boundary=boundary,valid_depth=arrays['inputs'][2]>0)
                    if 'original_valid' in arrays: regions['natural_holes']=~arrays['original_valid']
                    if 'added' in arrays: regions['added_holes']=arrays['added']
                    row['region_confusions']={}
                    for key,mask in regions.items():
                        mask=mask & ok
                        row['region_confusions'][key]=np.bincount(y[mask]*classes+pred[mask],minlength=classes*classes).reshape(classes,classes).tolist()
                row['ca_spatial_std_mean'] = float(m.maps['ca_effective'].float().flatten(-2).std(-1,unbiased=False).mean())
                if domain=='private':
                    row.update(primary_class=meta[int(sample)]['primary_class'], foreground_area=int((y==1).sum()))
                resized = {k: F.interpolate(m.maps[k].float(), y.shape, mode='bilinear', align_corners=False)[0,0].cpu().numpy()
                           for k in ('consistency', 'gate_d_to_i', 'gate_i_to_d')}
                row['quadrants'] = {}
                for q in range(4):
                    mask = ok & (quadrant == q)
                    row['quadrants'][str(q)] = dict(pixels=int(mask.sum()), fused_correct=int(((pred==y)&mask).sum()),
                        **{k: float(v[mask].mean()) if mask.any() else None for k,v in resized.items()})
                row['alpha'] = float(.5+.5*m.alpha.clamp(0,1)) if m.arm in ('P','PG') else 0.
                row['threshold'] = float(m.threshold)
                records.append(row)
                if idx in chosen:
                    for k,v in m.maps.items():
                        if k in ('ca_local','ca_effective'): continue  # Reconstruct exactly from grid/global/alpha.
                        arrays[f's{stage}_{k}'] = v.float().cpu().numpy().astype(np.float16)
                    arrays[f's{stage}_alpha'] = np.float32(row['alpha'])
                    for tag,z in zip(('i','d'), m.logits):
                        arrays[f's{stage}_prob_{tag}'] = z.float().softmax(1)[0].cpu().numpy().astype(np.float16)
                    arrays[f's{stage}_quadrants'] = quadrant
                    change = np.zeros(y.shape,dtype=np.uint8)
                    change[(quadrant==0)&(pred==y)&ok] = 1
                    change[(quadrant==3)&(pred!=y)&ok] = 2
                    arrays[f's{stage}_correction'] = change
                    panels.extend([(f'S{stage} I-side probe', pi, False), (f'S{stage} D-side probe', pd, False),
                                   (f'S{stage} agreement', resized['consistency'], True),
                                   (f'S{stage} D->I gate', resized['gate_d_to_i'], True),
                                   (f'S{stage} I->D gate', resized['gate_i_to_d'], True)])
                    panels.append((f'S{stage} rescued/damaged', change, 'correction'))
                    for tag,z in zip(('I','D'),ps):
                        ent=-(z*z.clamp_min(1e-8).log()).sum(0)/np.log(classes)
                        panels.append((f'S{stage} {tag} entropy',ent.cpu().numpy(),True))
                    # Fixed channel identities, never pick channels by favorable output.
                    for channel in (0, 1, 2):
                        ca = m.maps['ca_effective'][0,0,channel].float().cpu().numpy()
                        if ca.shape == (1,1): ca = np.broadcast_to(ca, m.maps['consistency'].shape[-2:])
                        panels.append((f'S{stage} I-CA ch{channel}', ca, True))
            if idx in chosen:
                dest = root/condition; dest.mkdir(exist_ok=True)
                np.savez_compressed(dest/f'frame_{idx:04d}.npz', **arrays)
                sheet(dest/f'frame_{idx:04d}.png', panels)
                save_json(dest/f'frame_{idx:04d}.json', dict(sample_id=sample, index=idx))
            for m in model.backbone.FRMs:
                m.capture = False; m.maps = {}
        if full:
            np.savez_compressed(root/f'{condition}_predictions.npz',sample_ids=np.asarray(all_ids),pred=np.stack(all_masks))
    save_json(root/'statistics.json', dict(rows=records, conditions=conditions,
        quadrant_codes={'0':'both_wrong', '1':'D_only_correct', '2':'I_only_correct', '3':'both_correct'},
        note='Stages >0 already contain previous FRM exchange. Probes are detached auxiliary heads, not unimodal networks.'))
    return conditions
