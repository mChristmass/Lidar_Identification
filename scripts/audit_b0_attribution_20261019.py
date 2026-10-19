"""Inference-only CMX-B0 failure attribution audit (development sets only)."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.io import loadmat
from PIL import Image
from experiments.paper_benchmark_v1.data import PrivatePaperDataset,robust_depth
from experiments.paper_benchmark_v1.corruptions_v2 import corrupt
from experiments.paper_benchmark_v1.export_group_meeting_visuals import image_score
from experiments.paper_benchmark_v1.paired_model import PairedCMX
from experiments.paper_benchmark_v1.prepare_nyuv2 import image_for_index,depth_for_index,labels_for_index

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def boundary(y):
    ok=y!=255;o=np.zeros(y.shape,bool);h=(y[:,1:]!=y[:,:-1])&ok[:,1:]&ok[:,:-1];v=(y[1:]!=y[:-1])&ok[1:]&ok[:-1];o[:,1:]|=h;o[:,:-1]|=h;o[1:]|=v;o[:-1]|=v;return o
def components(y,classes):
    out=[];ok=y!=255
    for c in ([1] if classes==2 else np.unique(y[ok]).tolist()):
        lab,_=ndimage.label(y==c);out += [int(x) for x in np.bincount(lab.ravel())[1:] if x>=5]
    return out
def private_items(root):
    d=PrivatePaperDataset(root,Path(root)/'paper_split_v1/dev_indices.npy');return [d[i] for i in range(len(d))]
def nyu_items(raw):
    import h5py
    raw=Path(raw);labs=loadmat(raw/'labels40.mat')['labels40']; rows=read('data/paper_benchmark_runs/paired_window_v1/nyuv2/B0_seed42/curves/clean_medium.json')['degradation_audit'];out=[]
    with h5py.File(raw/'nyu_depth_v2_labeled.mat','r') as f:
        for q in rows:
            sid=q['sample_id'];i=int(sid)-1;rgb=image_for_index(f['images'],i);a=np.rint(.299*rgb[...,0]+.587*rgb[...,1]+.114*rgb[...,2]).astype(np.uint8);d=depth_for_index(f['rawDepths'],i);a,d,v,_=corrupt(a,d,'clean','medium',sid);out.append(dict(sample_id=sid,inputs=torch.from_numpy(np.stack((a,robust_depth(d,v),v.astype(np.float32)))),labels=torch.from_numpy(labels_for_index(labs,i).astype(np.int64)),raw_i=np.rint(.299*rgb[...,0]+.587*rgb[...,1]+.114*rgb[...,2]).astype(np.uint8),raw_d=depth_for_index(f['rawDepths'],i)))
    return out
def model(path,variant,classes,size,device):
    m=PairedCMX('B0',variant,classes,size,256).to(device).eval();s=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(s['model'],strict=True);return m
@torch.inference_mode()
def pred(m,x): return m(x[None].to(next(m.parameters()).device))['logits'].argmax(1)[0].cpu().numpy().astype(np.uint8)
def regions(y,x):
    ok=y!=255;b=boundary(y);return {'all':ok,'boundary':b&ok,'valid_depth':(x[2]>.5)&ok,'invalid_depth':(x[2]<=.5)&ok}
def process(domain,items,m,classes,out):
    rows=[];out=Path(out)/domain;out.mkdir(parents=True,exist_ok=True)
    for n,b in enumerate(items,1):
        x=b['inputs'].float();y=b['labels'].numpy();full=pred(m,x);xi=x.clone();xi[0]=0;pd=pred(m,xi);xd=x.clone();xd[1:]=0;pi=pred(m,xd);ok=y!=255
        def sc(z):return float(image_score(z,y,classes))
        low=F.interpolate(F.avg_pool2d(x[None],2,2),size=x.shape[-2:],mode='bilinear',align_corners=False)[0];pl=pred(m,low)
        rr=regions(y,x.numpy()); fs=sc(full);iscore=sc(pi);dscore=sc(pd);r=dict(sample_id=str(b['sample_id']),full_iou=fs,i_only_iou=iscore,d_only_iou=dscore,lowres_iou=sc(pl),scale_drop=fs-sc(pl),intensity_rescue=iscore-dscore,depth_rescue=dscore-iscore,modal_conflict=float(((pi!=pd)&ok).mean()),conflict_hurts_full=bool(fs+1e-9<max(iscore,dscore)),full_vs_i_disagreement=float((full[ok]!=pi[ok]).mean()),full_vs_d_disagreement=float((full[ok]!=pd[ok]).mean()),hole_fraction=float((x[2].numpy()<=.5).mean()),target_hole_fraction=float((x[2].numpy()[y==1]<=.5).mean()) if classes==2 and (y==1).any() else None,boundary_fraction=float(rr['boundary'].sum()/max(ok.sum(),1)),smallest_component=min(components(y,classes) or [None]),gt_area=int((y==1).sum()) if classes==2 else None,region_accuracy={k:float((full[v]==y[v]).mean()) if v.any() else None for k,v in rr.items()});rows.append(r)
        if n%20==0:print(f'B0_CASES domain={domain} {n}/{len(items)}',flush=True)
    valid=[r for r in rows if r['gt_area'] is None or r['gt_area']>0]
    sel={'failure':sorted(valid,key=lambda r:r['full_iou'])[:10],'modal_conflict':sorted(valid,key=lambda r:(r['conflict_hurts_full'],r['modal_conflict']),reverse=True)[:10],'depth_hole_boundary':sorted(valid,key=lambda r:r['hole_fraction']*r['boundary_fraction'],reverse=True)[:10],'small_or_thin':sorted(valid,key=lambda r:(r['smallest_component'] if r['smallest_component'] is not None else 10**9))[:10],'scale_loss':sorted(valid,key=lambda r:r['scale_drop'],reverse=True)[:10],'intensity_vs_depth':sorted(valid,key=lambda r:abs(r['intensity_rescue']-r['depth_rescue']),reverse=True)[:10]}
    save(out/'cases.json',dict(domain=domain,rows=rows,selections={k:[r['sample_id'] for r in v] for k,v in sel.items()},definitions={'i_only':'depth+valid zero, intensity retained','d_only':'intensity zero, depth+valid retained','no_clean':'observed data only; no clean target'}));return rows
def private_shift(root,out):
    root=Path(root);I=np.load(root/'intensity.npy',mmap_mode='r');D=np.load(root/'depth.npy',mmap_mode='r');
    def stats(ids):
        q=[]
        for z in ids:
            i=np.asarray(I[int(z)-1],np.float32);d=np.asarray(D[int(z)-1],np.float32);v=np.isfinite(d)&(d>0);q.append([float(np.nanmean(i)),float(np.nanstd(i)),float(v.mean()),float(np.nanmean(d[v])) if v.any() else 0])
        a=np.asarray(q);return {'count':len(q),'mean':a.mean(0).tolist(),'std':a.std(0).tolist()}
    save(Path(out)/'distribution_shift.json',{'private':{'train':stats(np.load(root/'paper_split_v1/train_indices.npy')),'dev':stats(np.load(root/'paper_split_v1/dev_indices.npy'))},'note':'correlational train/dev shift; not causal evidence'})
def nyu_stress(items,m,out):
    """Matched public counterfactuals: clean, photon, holes and joint."""
    result=[]
    for n,b in enumerate(items,1):
        sid=b['sample_id'];y=b['labels'].numpy();base=b['inputs'];base_score=float(image_score(pred(m,base),y,40));row={'sample_id':sid,'clean_iou':base_score}
        for family in ('photon_proxy','block_missing','joint'):
            a,d,v,added=corrupt(b['raw_i'],b['raw_d'],family,'heavy',sid);x=torch.from_numpy(np.stack((a,robust_depth(d,v),v.astype(np.float32))));p=pred(m,x);score=float(image_score(p,y,40));ok=y!=255;hole=added&ok;row[family]={'iou':score,'drop':base_score-score,'added_hole_fraction':float(added.mean()),'added_hole_accuracy':float((p[hole]==y[hole]).mean()) if hole.any() else None,'non_added_valid_accuracy':float((p[(~added)&(v>0)&ok]==y[(~added)&(v>0)&ok]).mean()) if ((~added)&(v>0)&ok).any() else None}
        result.append(row)
        if n%20==0:print(f'B0_STRESS nyuv2 {n}/{len(items)}',flush=True)
    save(Path(out)/'nyuv2_stress.json',{'rows':result,'selection':{'photon_largest_drop':sorted(result,key=lambda r:r['photon_proxy']['drop'],reverse=True)[:10],'hole_largest_drop':sorted(result,key=lambda r:r['block_missing']['drop'],reverse=True)[:10],'joint_largest_drop':sorted(result,key=lambda r:r['joint']['drop'],reverse=True)[:10]},'note':'Artificial stress proxies, not sensor-physics ground truth'})
def main():
    p=argparse.ArgumentParser();p.add_argument('--private-root',type=Path,default=Path('data/new_data/merged'));p.add_argument('--nyu-raw',type=Path,default=Path('data/public_semseg/nyuv2/raw'));p.add_argument('--private-checkpoint',type=Path,required=True);p.add_argument('--nyu-checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('reports/b0_attribution_20261019'));p.add_argument('--device',default='cuda');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);dev=torch.device(a.device);torch.set_grad_enabled(False)
    process('private',private_items(a.private_root),model(a.private_checkpoint,'CMX_B1',2,128,dev),2,a.output);torch.cuda.empty_cache() if dev.type=='cuda' else None
    nm=model(a.nyu_checkpoint,'CMX_B2',40,640,dev);nyu=nyu_items(a.nyu_raw);process('nyuv2',nyu,nm,40,a.output);nyu_stress(nyu,nm,a.output);private_shift(a.private_root,a.output);save(a.output/'protocol.json',{'scope':'dev-only inference; seed42 B0; no final-test access','created':time.time(),'note':'Zeroing and synthetic stress are sensitivity evidence, not clean reconstruction or unique physical causality'});print('B0_ATTRIBUTION_COMPLETE',flush=True)
if __name__=='__main__':main()
