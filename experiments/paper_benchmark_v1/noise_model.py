"""CMX intensity-noise controls, initialized to the original network."""
import torch
from torch import nn
from torch.nn import functional as F

from .model import CMXPaper


class NoiseCMX(CMXPaper):
    def __init__(self, method, backbone='CMX_B2', num_classes=40,
                 image_size=640, decoder_dim=256):
        super().__init__(backbone, num_classes, image_size, decoder_dim)
        if method not in ('C1', 'D1', 'D2', 'S1', 'S2'):
            raise ValueError(method)
        self.method = method
        if method != 'C1':
            self.restorer = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.GELU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.GELU(),
                nn.Conv2d(16, 1, 3, padding=1))
            nn.init.zeros_(self.restorer[-1].weight)
            nn.init.zeros_(self.restorer[-1].bias)
        if method == 'S2':
            self.noise_gate = nn.Sequential(
                nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(8, 8, 3, stride=2, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d(1), nn.Conv2d(8, 1, 1))
            nn.init.zeros_(self.noise_gate[-1].weight)
            nn.init.constant_(self.noise_gate[-1].bias, -2.)
        if method == 'D2':
            self.reliability = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.GELU(),
                nn.Conv2d(16, 1, 1))
            nn.init.zeros_(self.reliability[-1].weight)
            nn.init.constant_(self.reliability[-1].bias, 2.)
            self.fusion_strength = nn.Parameter(torch.zeros(4))

    def load_baseline(self, state):
        expected = {k for k in self.state_dict()
                    if k.startswith(('backbone.', 'decode_head.'))}
        if set(state) != expected:
            raise ValueError('Baseline state does not exactly match CMX')
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(k in expected for k in result.missing_keys):
            raise ValueError(result)

    @torch.no_grad()
    def project_parameters(self):
        if self.method == 'D2':
            self.fusion_strength.clamp_(0., 1.)

    def restore(self, inputs):
        intensity = inputs[:, :1]
        delta = self.restorer(inputs)
        gate_logits = torch.zeros_like(intensity[:, :, :1, :1])
        gate = torch.ones_like(gate_logits)
        if self.method == 'S2':
            features = torch.cat([intensity] + [
                (intensity - F.avg_pool2d(F.pad(intensity, (k//2,)*4,
                                               mode='replicate'), k, 1)).abs()
                for k in (3, 7)], dim=1)
            gate_logits = self.noise_gate(features)
            gate = gate_logits.sigmoid()
        return {'restored_intensity': (intensity + gate*delta).clamp(0., 1.),
                'raw_restored_intensity': (intensity + delta).clamp(0., 1.),
                'noise_gate_logits': gate_logits, 'noise_gate': gate}

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError('Expected intensity, normalized depth, validity')
        intensity = inputs[:, :1]
        if self.method == 'C1':
            return super().forward(inputs)
        # The correction and confidence both use observed inputs only.
        restoration = self.restore(inputs)
        corrected = restoration['restored_intensity']
        corrected_inputs = torch.cat((corrected, inputs[:, 1:]), dim=1)
        if self.method in ('D1', 'S1', 'S2'):
            out = super().forward(corrected_inputs)
            out.update(restoration)
            return out
        confidence = self.reliability(inputs).sigmoid()
        a, d = corrected, inputs[:, 1:2]
        b = self.backbone
        outputs = []
        for stage in range(1, 5):
            a, h, w = getattr(b, f'patch_embed{stage}')(a)
            d, _, _ = getattr(b, f'extra_patch_embed{stage}')(d)
            for block in getattr(b, f'block{stage}'):
                a = block(a, h, w)
            for block in getattr(b, f'extra_block{stage}'):
                d = block(d, h, w)
            a = getattr(b, f'norm{stage}')(a)
            d = getattr(b, f'extra_norm{stage}')(d)
            a = a.reshape(a.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            d = d.reshape(d.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            a, d = b.FRMs[stage - 1](a, d)
            ffm = b.FFMs[stage - 1]
            a_tokens = a.flatten(2).transpose(1, 2)
            d_tokens = d.flatten(2).transpose(1, 2)
            cross_a, cross_d = ffm.cross(a_tokens, d_tokens)
            conf = F.adaptive_avg_pool2d(confidence, (h, w))
            attenuation = (self.fusion_strength[stage - 1] * (1 - conf)).flatten(2).transpose(1, 2)
            cross_d = cross_d - attenuation * (cross_d - d_tokens)
            outputs.append(ffm.channel_emb(torch.cat((cross_a, cross_d), dim=-1), h, w))
        logits = self.decode_head(outputs)
        return {'logits': F.interpolate(logits, inputs.shape[-2:], mode='bilinear',
                                        align_corners=False),
                'restored_intensity': corrected, 'intensity_confidence': confidence}
