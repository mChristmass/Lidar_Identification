"""Render the cited CMX-B0 dev examples and their matched counterfactuals."""
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from experiments.paper_benchmark_v1.corruptions_v2 import corrupt
from experiments.paper_benchmark_v1.data import robust_depth
from experiments.paper_benchmark_v1.export_group_meeting_visuals import make_panel,gray,label_image
from scripts.audit_b0_attribution_20261019 import private_items,nyu_items,model,pred

OUT=Path('reports/b0_attribution_20261019/figures');OUT.mkdir(parents=True,exist_ok=True)
PRIVATE={
 'private_shared_misses':['436','451','958','959','960','962'],
 'private_scale':['421','443','963','454','1132'],
 'private_modal':['400','393','966','949'],
 'private_holes_boundary':['1147','1153','1137','1146'],
}
NYU={
 'nyu_clean_failures':['0013','0507','0162','0095','1172'],
 'nyu_modal':['0162','0897','0968','1378','0013'],
 'nyu_photon':['1311','0875','1105','0020','0825'],
 'nyu_holes':['0024','0376','0602','1425','0806'],
 'nyu_joint':['0703','0722','1311','1105','0020'],
}
torch.set_num_threads(4)

def panel(groups,items,m,classes,domain):
    for name,ids in groups.items():
        rows=[];names=[]
        for sid in ids:
            b=items[sid];x=b['inputs'].float();y=b['labels'].numpy();full=pred(m,x)
            d=x.clone();d[1:]=0; i=x.clone();i[0]=0
            preds=[full,pred(m,d),pred(m,i)]
            if domain=='private':
                low=F.interpolate(F.avg_pool2d(x[None],2,2),size=x.shape[-2:],mode='bilinear',align_corners=False)[0]
                low_pred=pred(m,low)
                xy=np.argwhere(y==1)
                if len(xy):
                    lo=np.maximum(xy.min(0)-12,0);hi=np.minimum(xy.max(0)+13,y.shape);crop=np.s_[lo[0]:hi[0],lo[1]:hi[1]]
                else:crop=np.s_[:,:]
                cols=['I','D','valid','GT','B0','I retained','D retained','2x pooled B0']
                rows.append([gray(x[0][crop]),gray(x[1][crop]),gray(x[2][crop]),label_image(y[crop],classes)]+[label_image(z[crop],classes) for z in preds]+[label_image(low_pred[crop],classes)]);names.append(f'{sid}\nGT+12px crop')
            else:
                cols=['I','D','valid','GT','B0','I retained','D retained']
                rows.append([gray(x[0]),gray(x[1]),gray(x[2]),label_image(y,classes)]+[label_image(z,classes) for z in preds]);names.append(sid)
        make_panel(rows,cols,names,OUT/f'{name}.png',cell=(170,145) if domain=='private' else (210,160))
        print('FIGURE',name,flush=True)

def stress(items,m):
    for name,family in [('nyu_photon','photon_proxy'),('nyu_holes','block_missing'),('nyu_joint','joint')]:
        rows=[];names=[]
        for sid in NYU[name]:
            b=items[sid];y=b['labels'].numpy();x=b['inputs'].float()
            a,d,v,added=corrupt(b['raw_i'],b['raw_d'],family,'heavy',sid)
            bad=torch.from_numpy(np.stack([a,robust_depth(d,v),v.astype(np.float32)]))
            cols=['Clean I','Clean D','GT','B0 clean','Degraded I','Degraded D','Added holes','B0 degraded']
            rows.append([gray(x[0]),gray(x[1]),label_image(y,40),label_image(pred(m,x),40),gray(bad[0]),gray(bad[1]),gray(added),label_image(pred(m,bad),40)])
            names.append(sid)
        make_panel(rows,cols,names,OUT/f'{name}_paired.png',cell=(190,145))
        print('FIGURE',name,'paired',flush=True)

def main():
    private={b['sample_id']:b for b in private_items('data/new_data/merged')}
    pm=model('data/paper_benchmark_runs/paired_window_v1/private/B0_seed42/last.pth','CMX_B1',2,128,'cpu')
    panel(PRIVATE,private,pm,2,'private');del pm
    nyu={b['sample_id']:b for b in nyu_items('data/public_semseg/nyuv2/raw')}
    nm=model('data/paper_benchmark_runs/paired_window_v1/nyuv2/B0_seed42/last.pth','CMX_B2',40,640,'cpu')
    panel({k:v for k,v in NYU.items() if k in ('nyu_clean_failures','nyu_modal')},nyu,nm,40,'nyuv2')
    stress(nyu,nm)
    print('B0_FIGURES_COMPLETE',flush=True)
if __name__=='__main__':main()
