"""Matched combinations of task-driven correction and semantic detail refinement."""
import torch
from torch import nn
from torch.nn import functional as F
from .paired_model import PairedCMX
from .broad_window_model import BroadCMX, backward_objective
from .structure_model import conv_block, zero_output

COMBINATIONS = ('RD_CORRECTED', 'RD_RAW')


class RepairDetailCMX(PairedCMX):
    def __init__(self, method, variant='CMX_B2', num_classes=40, image_size=640, decoder_dim=256):
        if method not in COMBINATIONS:
            raise ValueError(method)
        super().__init__('PAIR',variant,num_classes,image_size,decoder_dim)
        self.combination_method = method
        self.candidate_enabled = True
        self.repair_enabled = True
        self.detail_enabled = True
        # Initialize each component exactly like its standalone historical candidate.
        # Re-entering the fork is intentional: the two historical constructors each
        # started from the SAME derived seed, not successive draws from one stream.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed((torch.initial_seed()+4919) % (2**63))
            repair=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.GELU(),
                nn.Conv2d(16,16,3,padding=1),nn.GELU(),nn.Conv2d(16,1,3,padding=1))
            nn.init.zeros_(repair[-1].weight)
            nn.init.zeros_(repair[-1].bias)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed((torch.initial_seed()+4919) % (2**63))
            detail=nn.ModuleDict({
                'stem':nn.Sequential(conv_block(3,16,2),conv_block(16,16)),
                'refine':nn.Sequential(conv_block(16+num_classes,16),zero_output(16,num_classes))})
        self.adapters['candidate']=nn.ModuleDict({'repair':repair,'detail':detail})

    def forward(self,inputs):
        if not self.adapters_enabled or not self.candidate_enabled:
            return super().forward(inputs)
        parts=self.adapters['candidate']
        delta=parts['repair'](inputs) if self.repair_enabled else torch.zeros_like(inputs[:,:1])
        corrected=torch.cat(((inputs[:,:1]+delta).clamp(0,1),inputs[:,1:]),1)
        result=super().forward(corrected)
        result['correction_penalty']=delta.float().square().mean()
        result['correction_mae']=(corrected[:,:1]-inputs[:,:1]).detach().abs().mean()
        if self.detail_enabled:
            # Both receive semantic context from the corrected-input backbone.
            # RD_RAW retains original observed detail; it is NOT clean ground truth.
            detail_input=inputs if self.combination_method=='RD_RAW' else corrected
            feature=parts['detail']['stem'](detail_input)
            semantic=F.interpolate(result['logits'],feature.shape[-2:],mode='bilinear',align_corners=False).softmax(1)
            residual=parts['detail']['refine'](torch.cat((feature,semantic),1))
            residual=F.interpolate(residual,inputs.shape[-2:],mode='bilinear',align_corners=False)
            result['logits']=result['logits']+residual
            result['detail_logit_rms']=residual.detach().float().square().mean().sqrt()
        return result


def build_model(method,*args,**kwargs):
    if method in ('R_RES','F_DETAIL'):
        return BroadCMX(method,*args,**kwargs)  # Exact historical implementation.
    return RepairDetailCMX(method,*args,**kwargs)
