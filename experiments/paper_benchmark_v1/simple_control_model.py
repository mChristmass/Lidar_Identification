"""Predeclared simple alternatives, trained under the unchanged PAIR protocol."""
import torch
from torch import nn
from torch.nn import functional as F
from .broad_window_model import BroadCMX, backward_objective
from .paired_model import PairedCMX
from .structure_model import conv_block, zero_output

METHODS=('FIX_SMOOTH','D_CONV','SMOOTH_DETAIL')


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def plain_count(width,classes):
    # Three 3x3 conv/GN blocks, then 1x1 output. GN has affine scale/bias.
    return 18*width*width+(36+classes)*width+classes


class SimpleControlCMX(BroadCMX):
    def __init__(self,method,variant='CMX_B2',num_classes=40,image_size=640,decoder_dim=256):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__('F_DETAIL',variant,num_classes,image_size,decoder_dim)
        self.control_method=method
        self.detail_reference_parameters=count_parameters(self.adapters['candidate'])
        if method=='FIX_SMOOTH':
            self.adapters.pop('candidate')  # No unused trainable detail parameters.
        elif method=='D_CONV':
            self.plain_width=min(range(4,129,4),key=lambda w:abs(plain_count(w,num_classes)-self.detail_reference_parameters))
            w=self.plain_width
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed((torch.initial_seed()+4919) % (2**63))
                self.adapters['candidate']=nn.Sequential(conv_block(3,w,2),conv_block(w,w),
                    conv_block(w,w),zero_output(w,num_classes))
        one=torch.tensor([1.,2.,1.])
        self.register_buffer('smooth_kernel',(one[:,None]*one[None,:]).reshape(1,1,3,3)/16)

    def smooth_inputs(self,x):
        # Fixed binomial Gaussian approximation, 50% blend; no labels/dev tuning.
        intensity=x[:,:1]
        blurred=F.conv2d(F.pad(intensity,(1,1,1,1),mode='replicate'),self.smooth_kernel.to(intensity))
        return torch.cat((.5*intensity+.5*blurred,x[:,1:]),1)

    def forward(self,x):
        if not self.adapters_enabled or not self.candidate_enabled:
            return PairedCMX.forward(self,x)
        if self.control_method=='D_CONV':
            result=PairedCMX.forward(self,x)
            # Ordinary I/D/M convolutional residual; no semantic-probability input.
            residual=F.interpolate(self.adapters['candidate'](x),x.shape[-2:],mode='bilinear',align_corners=False)
            result['logits']=result['logits']+residual
            result['detail_logit_rms']=residual.detach().float().square().mean().sqrt()
            return result
        smoothed=self.smooth_inputs(x)
        result=(super().forward(smoothed) if self.control_method=='SMOOTH_DETAIL'
                else PairedCMX.forward(self,smoothed))
        result['correction_mae']=(smoothed[:,:1]-x[:,:1]).detach().abs().mean()
        return result

    def control_metadata(self):
        candidate=count_parameters(self.adapters['candidate']) if 'candidate' in self.adapters else 0
        return dict(control=self.control_method,candidate_parameters=candidate,
            detail_reference_parameters=self.detail_reference_parameters,
            candidate_to_detail_parameter_ratio=candidate/self.detail_reference_parameters,
            plain_width=getattr(self,'plain_width',None),
            smoothing=(None if self.control_method=='D_CONV' else
                '3x3 binomial kernel /16, blend0.5; fixed before all training and evaluation views'),
            note='Approximate parameter control, not exact parameter/FLOP matching; all candidates active at inference')
