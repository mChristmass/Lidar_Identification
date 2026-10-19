"""Matched NYUv2 dev comparison for one frozen original CMX B0 checkpoint."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from experiments.paper_benchmark_v1.corruptions_v2 import corrupt
from experiments.paper_benchmark_v1.data import robust_depth
from scripts.audit_b0_attribution_20261019 import nyu_items,model,pred

CONDITIONS=('clean','photon_proxy','block_missing','joint')
MODES=('both','I_retained','D_retained')
def miou(matrix):
    u=matrix.sum(0)+matrix.sum(1)-matrix.diagonal();ok=u>0
    return float(np.mean(matrix.diagonal()[ok]/u[ok]))
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='data/paper_benchmark_runs/paired_window_v1/nyuv2/B0_seed42/last.pth');p.add_argument('--raw',default='data/public_semseg/nyuv2/raw');p.add_argument('--output',default='reports/b0_attribution_20261019/condition_modalities');p.add_argument('--device',default='cpu');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    items=nyu_items(a.raw);m=model(a.checkpoint,'CMX_B2',40,640,a.device)
    manifest={'checkpoint':a.checkpoint,'dataset':'NYUv2 frozen dev 80','conditions':list(CONDITIONS),'modes':list(MODES),'note':'Zero-input sensitivity of one frozen dual-modal model. I_retained: D=0; D_retained: I=0. B0 ignores validity mask.'}
    mp=out/'protocol.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise RuntimeError('Protocol changed; use a new output directory')
    mp.write_text(json.dumps(manifest,indent=2))
    for c in CONDITIONS:
        for mode in MODES:
            cp=out/f'{c}_{mode}.json'
            if cp.exists():continue
            confusion=np.zeros((40,40),np.int64);rows=[]
            for k,b in enumerate(items,1):
                sid=b['sample_id'];y=b['labels'].numpy();ok=y!=255
                if c=='clean':x=b['inputs'].clone()
                else:
                    ia,dep,valid,added=corrupt(b['raw_i'],b['raw_d'],c,'heavy',sid)
                    x=torch.from_numpy(np.stack((ia,robust_depth(dep,valid),valid.astype(np.float32))))
                if mode=='I_retained':x[1:]=0
                elif mode=='D_retained':x[0]=0
                z=pred(m,x)
                mat=np.bincount(y[ok]*40+z[ok],minlength=1600).reshape(40,40)
                confusion+=mat
                rows.append({'sample_id':sid,'miou':miou(mat),'pixel_accuracy':float((z[ok]==y[ok]).mean())})
                if k%20==0:print(f'B0_CONDITION {c} {mode} {k}/80',flush=True)
            cp.write_text(json.dumps({'condition':c,'mode':mode,'miou':miou(confusion),'pixel_accuracy':float(confusion.trace()/confusion.sum()),'confusion_matrix':confusion.tolist(),'rows':rows},indent=2))
    table=[];group={}
    for c in CONDITIONS:
        group[c]={mode:json.loads((out/f'{c}_{mode}.json').read_text()) for mode in MODES}
        table.append({'condition':c,**{mode:group[c][mode]['miou'] for mode in MODES},'gain_over_best_probe':group[c]['both']['miou']-max(group[c]['I_retained']['miou'],group[c]['D_retained']['miou'])})
    clean=group['clean'];drops=[]
    for c in CONDITIONS[1:]:
        g=group[c];drops.append({'condition':c,**{mode:clean[mode]['miou']-g[mode]['miou'] for mode in MODES}})
    result={'aggregate_miou':table,'clean_to_heavy_drop':drops,'note':'Dataset mIoU from pooled 40-class confusion matrices. Input zeroing is not separately trained single-modal performance.'}
    (out/'comparison.json').write_text(json.dumps(result,indent=2))
    print('B0_MODALITY_COMPARISON_COMPLETE',out,flush=True)
if __name__=='__main__':main()
