"""Semantic-supervised candidate screen on PAIR; no clean restoration target."""
import torch
from torch import nn
from torch.nn import functional as F
from .paired_model import PairedCMX, perturb_view, safe_ce
from .structure_model import conv_block, zero_output

METHODS = ('R_RES','R_FILTER','F_DETAIL')
CORRECTION_WEIGHT = .01


class AdaptiveFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.predict = nn.Sequential(conv_block(3,16,2),conv_block(16,16),zero_output(16,4))
        with torch.no_grad():
            self.predict[-1].bias[0] = 6.  # Near identity, NOT exact identity.

    def forward(self,x):
        intensity = x[:,:1]
        basis = [intensity]
        for size in (3,7,15):
            basis.append(F.avg_pool2d(F.pad(intensity,(size//2,)*4,mode='replicate'),size,1))
        weights = F.interpolate(self.predict(x),intensity.shape[-2:],mode='bilinear',align_corners=False).float().softmax(1)
        return (torch.cat(basis,1).float()*weights).sum(1,keepdim=True),weights


class BroadCMX(PairedCMX):
    def __init__(self,method,variant='CMX_B2',num_classes=40,image_size=640,decoder_dim=256):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__('PAIR',variant,num_classes,image_size,decoder_dim)
        self.broad_method = method
        self.candidate_enabled = True
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed((torch.initial_seed()+4919) % (2**63))
            if method == 'R_RES':
                self.adapters['candidate'] = nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.GELU(),
                    nn.Conv2d(16,16,3,padding=1),nn.GELU(),nn.Conv2d(16,1,3,padding=1))
                nn.init.zeros_(self.adapters['candidate'][-1].weight)
                nn.init.zeros_(self.adapters['candidate'][-1].bias)
            elif method == 'R_FILTER':
                self.adapters['candidate'] = AdaptiveFilter()
            elif method == 'F_DETAIL':
                self.adapters['candidate'] = nn.ModuleDict({
                    'stem': nn.Sequential(conv_block(3,16,2),conv_block(16,16)),
                    'refine': nn.Sequential(conv_block(16+num_classes,16),zero_output(16,num_classes))})

    def forward(self,inputs):
        if not self.adapters_enabled or not self.candidate_enabled:
            return super().forward(inputs)
        candidate = self.adapters['candidate']
        extra = {}
        if self.broad_method in ('R_RES','R_FILTER'):
            if self.broad_method=='R_RES':
                delta = candidate(inputs)
                corrected = (inputs[:,:1]+delta).clamp(0,1)
                penalty = delta.float().square().mean()
            else:
                corrected,weights = candidate(inputs)
                penalty = (corrected-inputs[:,:1]).float().square().mean()
                extra['identity_mass'] = weights[:,:1].detach().mean()
            result = super().forward(torch.cat((corrected,inputs[:,1:]),1))
            extra.update(correction_penalty=penalty,correction_mae=(corrected-inputs[:,:1]).detach().abs().mean())
        else:
            result = super().forward(inputs)
            detail = candidate['stem'](inputs)
            semantic = F.interpolate(result['logits'],detail.shape[-2:],mode='bilinear',align_corners=False).softmax(1)
            residual = candidate['refine'](torch.cat((detail,semantic),1))
            residual = F.interpolate(residual,inputs.shape[-2:],mode='bilinear',align_corners=False)
            result['logits'] = result['logits']+residual
            extra['detail_logit_rms'] = residual.detach().float().square().mean().sqrt()
        return dict(result,**extra)


def backward_objective(model,x,y,dataset,method,epoch,view_seed,scaler,weight):
    secondary,hidden = perturb_view(x,dataset,view_seed)
    stats = dict(loss=0.,ce_reference=0.,ce_secondary=0.,consistency=0.,completion=0.,
                 confidence_coverage=0.,hidden_fraction=float(hidden.float().mean()))
    for name,view in (('reference',x),('secondary',secondary)):
        with torch.autocast(device_type=x.device.type,enabled=x.device.type=='cuda'):
            pred = model(view)
            ce = safe_ce(pred['logits'],y)
            penalty = pred.get('correction_penalty',ce*0)
            objective = .5*(ce+CORRECTION_WEIGHT*penalty)
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite broad-screen objective')
        scaler.scale(objective*weight).backward()
        stats['loss'] += float(objective.detach())
        stats['ce_'+name] = float(ce.detach())
        for key in ('correction_penalty','correction_mae','identity_mass','detail_logit_rms'):
            if key in pred:
                stats[key] = stats.get(key,0.)+.5*float(pred[key].detach())
        del pred,objective,ce,penalty
    return stats
