"""Short screen: retain original fused features and add cross-scale task correction.

This is a feature-context hypothesis, not an image denoiser or established novelty.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .model import BACKBONES
from .structure_model import zero_output
from .task_control_model import TaskControlCMX


class ContextPyramid(nn.Module):
    def __init__(self, channels, width=32):
        super().__init__()
        self.project = nn.ModuleList([nn.Conv2d(c, width, 1) for c in channels])
        self.mix = nn.ModuleList([nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.GroupNorm(4, width), nn.GELU(), nn.Conv2d(width, width, 1), nn.GELU())
            for _ in channels])
        self.out = nn.ModuleList([zero_output(width, c) for c in channels])

    def forward(self, features):
        deltas = [None] * len(features)
        context = None
        for stage in reversed(range(len(features))):
            local = self.project[stage](features[stage])
            if context is not None:
                local = (local + F.interpolate(context, local.shape[-2:],
                         mode='bilinear', align_corners=False))/math.sqrt(2.)
            context = self.mix[stage](local)
            deltas[stage] = self.out[stage](context)
        return [f+d for f,d in zip(features,deltas)], deltas


class ContextCMX(TaskControlCMX):
    def __init__(self, method, variant='CMX_B2', num_classes=40, image_size=640, decoder_dim=256):
        if method not in ('B0','S_RAW','F_CTX','SF_CTX'):
            raise ValueError(method)
        super().__init__('S_RAW' if method in ('S_RAW','SF_CTX') else 'B0',
                         variant, num_classes, image_size, decoder_dim)
        self.context_method = method
        self.context_enabled = True
        if method in ('F_CTX','SF_CTX'):
            if not hasattr(self,'adapters'):
                self.adapters = nn.ModuleDict()
            # Match context-branch initialization between F_CTX and SF_CTX without
            # changing the existing S_RAW/base initialization or the caller RNG state.
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed((torch.initial_seed()+1701) % (2**63))
                self.adapters['context'] = ContextPyramid(BACKBONES[variant][1])
            # Both existing forward paths call the same decoder. This hook does not
            # change checkpoint key names, source weights, or the upstream encoder code.
            self.decode_head.register_forward_pre_hook(self._apply_context)

    def _apply_context(self, module, args):
        if not self.adapters_enabled or not self.context_enabled:
            return None
        features = args[0]
        changed, deltas = self.adapters['context'](features)
        if self.audit_enabled:
            for stage,(feature,delta) in enumerate(zip(features,deltas),1):
                f = feature.detach().float().square().mean().sqrt().item()
                d = delta.detach().float().square().mean().sqrt().item()
                self.audit_rows.append(dict(stage=f'context_{stage}',residual_rms=d,
                                           relative_rms=d/max(f,1e-8)))
        return (changed, *args[1:])


def task_loss(prediction, labels):
    # No clean target, pixel correction, teacher, reconstruction, or extra regularizer.
    return F.cross_entropy(prediction['logits'],labels,ignore_index=255)
