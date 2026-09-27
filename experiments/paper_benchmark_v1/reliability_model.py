"""Exploratory FRM residual modulation; no oracle corruption information."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import CMXPaper


class ReliabilityCMX(CMXPaper):
    def __init__(self, experiment, variant='CMX_B2', num_classes=40,
                 image_size=640, decoder_dim=256):
        super().__init__(variant, num_classes, image_size, decoder_dim)
        if experiment not in ('R1', 'R2', 'R3'):
            raise ValueError(experiment)
        self.experiment = experiment
        if experiment == 'R1':
            self.validity_projection = nn.Conv2d(1, 1, 1, bias=False)
            nn.init.zeros_(self.validity_projection.weight)
        else:
            self.depth_strength = nn.Parameter(torch.zeros(4))
            if experiment == 'R3':
                self.intensity_strength = nn.Parameter(torch.zeros(4))
                self.reliability = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1),
                                                 nn.GELU(), nn.Conv2d(8, 1, 1))
                nn.init.zeros_(self.reliability[-1].weight)
                nn.init.constant_(self.reliability[-1].bias, 2.)

    def load_baseline(self, state):
        expected = {k for k in self.state_dict() if k.startswith(('backbone.', 'decode_head.'))}
        if set(state) != expected:
            raise ValueError('Baseline keys differ from original CMX model')
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(k in expected for k in result.missing_keys):
            raise ValueError(result)

    @torch.no_grad()
    def project_parameters(self):
        for name in ('depth_strength', 'intensity_strength'):
            if hasattr(self, name):
                getattr(self, name).clamp_(0., 1.)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError('Expected intensity, normalized depth, observed validity')
        valid = inputs[:, 2:3].clamp(0, 1)
        if self.experiment == 'R1':
            depth = inputs[:, 1:2] + self.validity_projection(valid)
            return super().forward(torch.cat((inputs[:, :1], depth), dim=1))
        intensity, depth = inputs[:, :1], inputs[:, 1:2]
        confidence = None
        if self.experiment == 'R3':
            mean = F.avg_pool2d(intensity, 5, stride=1, padding=2, count_include_pad=False)
            evidence = torch.cat((intensity, mean, (intensity - mean).abs()), dim=1)
            # Segmentation-supervised confidence, NOT calibrated photon SNR.
            confidence = self.reliability(evidence).sigmoid()
        outputs = []
        b = self.backbone
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
            rect_a, rect_d = b.FRMs[stage - 1](a, d)
            support = F.adaptive_avg_pool2d(valid, (h, w))
            # Preserve semantic features; suppress only cross-modal FRM residuals.
            intensity = rect_a - self.depth_strength[stage - 1] * (1 - support) * (rect_a - a)
            depth = rect_d
            if confidence is not None:
                reliable = F.adaptive_avg_pool2d(confidence, (h, w))
                depth = rect_d - self.intensity_strength[stage - 1] * (1 - reliable) * (rect_d - d)
            # FFM remains unchanged; this is not full masked attention.
            outputs.append(b.FFMs[stage - 1](intensity, depth))
        logits = self.decode_head(outputs)
        return {'logits': F.interpolate(logits, inputs.shape[-2:], mode='bilinear', align_corners=False)}
