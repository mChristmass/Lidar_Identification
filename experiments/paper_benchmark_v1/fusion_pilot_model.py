"""Experimental validity-aware CMX fusion and invalid-depth feature propagation."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import CMXPaper


class FusionPilotCMX(CMXPaper):
    """F1 gates FFM cross residuals; F2 also propagates nearby depth features."""

    def __init__(self, variant_name, backbone='CMX_B2', num_classes=40,
                 image_size=640, decoder_dim=256):
        super().__init__(backbone, num_classes, image_size, decoder_dim)
        if variant_name not in ('F1', 'F2'):
            raise ValueError(variant_name)
        self.variant_name = variant_name
        self.cross_gate = nn.Parameter(torch.zeros(4))
        if variant_name == 'F2':
            self.propagation_gate = nn.Parameter(torch.zeros(4))

    def load_baseline(self, state):
        base_keys = {k for k in self.state_dict() if k.startswith(('backbone.', 'decode_head.'))}
        if set(state) != base_keys:
            raise ValueError('Source checkpoint does not exactly match CMX')
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(k in base_keys for k in result.missing_keys):
            raise ValueError(result)

    @torch.no_grad()
    def project_parameters(self):
        self.cross_gate.clamp_(0., 1.)
        if self.variant_name == 'F2':
            self.propagation_gate.clamp_(0., 1.)

    @staticmethod
    def propagate_depth(depth, support):
        # Normalize by nearby valid evidence; retain the original if none exists.
        mass = F.avg_pool2d(support, 5, stride=1, padding=2, count_include_pad=False)
        summed = F.avg_pool2d(depth * support, 5, stride=1, padding=2, count_include_pad=False)
        estimate = summed / mass.clamp_min(1e-5)
        return torch.where(mass > 1e-5, estimate, depth)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError('Expected intensity, normalized depth, observed validity')
        valid = inputs[:, 2:3].clamp(0, 1)
        intensity, depth = inputs[:, :1], inputs[:, 1:2]
        b = self.backbone
        outputs = []
        for stage in range(1, 5):
            a, h, w = getattr(b, f'patch_embed{stage}')(intensity)
            d, _, _ = getattr(b, f'extra_patch_embed{stage}')(depth)
            for block in getattr(b, f'block{stage}'):
                a = block(a, h, w)
            for block in getattr(b, f'extra_block{stage}'):
                d = block(d, h, w)
            a = getattr(b, f'norm{stage}')(a)
            d = getattr(b, f'extra_norm{stage}')(d)
            a = a.reshape(a.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            d = d.reshape(d.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            a, d = b.FRMs[stage - 1](a, d)
            support = F.adaptive_avg_pool2d(valid, (h, w))
            if self.variant_name == 'F2':
                estimate = self.propagate_depth(d, support)
                d = d + self.propagation_gate[stage - 1] * (1 - support) * (estimate - d)
            ffm = b.FFMs[stage - 1]
            a_tokens = a.flatten(2).transpose(1, 2)
            d_tokens = d.flatten(2).transpose(1, 2)
            cross_a, cross_d = ffm.cross(a_tokens, d_tokens)
            attenuation = (self.cross_gate[stage - 1] * (1 - support)).flatten(2).transpose(1, 2)
            cross_a = cross_a - attenuation * (cross_a - a_tokens)
            merged = torch.cat((cross_a, cross_d), dim=-1)
            outputs.append(ffm.channel_emb(merged, h, w))
            # The original CMX also carries FRM features into the next stage.
            intensity, depth = a, d
        logits = self.decode_head(outputs)
        return {'logits': F.interpolate(logits, inputs.shape[-2:], mode='bilinear', align_corners=False)}
