"""Observed support/structure descriptors, NOT photon uncertainty probabilities."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import CMXPaper, BACKBONES

DESCRIPTORS = ('valid', 'support3', 'support7', 'intensity_residual3',
               'intensity_residual7', 'depth_residual3', 'depth_residual7', 'depth_std5')


def local_mean(x, size):
    return F.avg_pool2d(F.pad(x, (size//2,)*4, mode='replicate'), size, 1)


def observation_descriptors(inputs):
    # FP32 avoids cancellation/overflow in mixed-precision local moments.
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        x = torch.nan_to_num(inputs.float()).clamp(0, 1)
        i, d, m = x[:, :1], x[:, 1:2], (x[:, 2:3]>.5).float()
        supports = [local_mean(m, k) for k in (3, 7)]
        ir = [(i-local_mean(i, k)).abs() for k in (3, 7)]
        dr = [m*(d-local_mean(d*m, k)/support.clamp_min(1e-6)).abs()
              for k, support in zip((3, 7), supports)]
        support5 = local_mean(m, 5)
        mean = local_mean(d*m, 5)/support5.clamp_min(1e-6)
        variance = (local_mean(d*d*m, 5)/support5.clamp_min(1e-6)-mean.square()).clamp_min(0)
        std = torch.where(support5>0, variance.sqrt(), torch.zeros_like(variance))
        return torch.cat([m, *supports, *ir, *dr, std], 1).clamp(0, 1)


class ObservationCMX(CMXPaper):
    def __init__(self, method, variant='CMX_B2', num_classes=40, image_size=640, decoder_dim=256):
        super().__init__(variant, num_classes, image_size, decoder_dim)
        if method not in ('B0', 'L', 'E', 'C'):
            raise ValueError(method)
        self.method = method
        if method != 'B0':
            self.adapters = nn.ModuleList()
            for channels in BACKBONES[variant][1]:
                if method == 'L':
                    adapter = nn.Sequential(nn.Conv2d(channels, 16, 1), nn.GELU(),
                                            nn.Conv2d(16, 16, 3, padding=1), nn.GELU(),
                                            nn.Conv2d(16, channels, 1))
                else:
                    adapter = nn.Sequential(nn.Conv2d(12 if method=='C' else 8, 16, 3, padding=1), nn.GELU(),
                                            nn.Conv2d(16, channels*(2 if method=='C' else 1), 1))
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
                self.adapters.append(adapter)

    def load_baseline(self, state):
        expected = {k for k in self.state_dict() if k.startswith(('backbone.', 'decode_head.'))}
        if set(state) != expected:
            raise ValueError('Baseline keys differ')
        self.load_state_dict(state, strict=False)

    def forward(self, inputs):
        if self.method == 'B0':
            return super().forward(inputs)
        desc = observation_descriptors(inputs) if self.method in ('E', 'C') else None
        if self.method != 'C':
            features = self.backbone(inputs[:, :1], inputs[:, 1:2])
            features = [f + adapter(f if self.method=='L' else
                        F.adaptive_avg_pool2d(desc, f.shape[-2:]))
                        for f, adapter in zip(features, self.adapters)]
        else:
            a, d = inputs[:, :1], inputs[:, 1:2]
            b, features = self.backbone, []
            for stage, adapter in enumerate(self.adapters, 1):
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
                a, d = b.FRMs[stage-1](a, d)
                context = torch.cat((F.adaptive_avg_pool2d(desc, (h, w)),
                                     a.mean(1,keepdim=True), a.amax(1,keepdim=True),
                                     d.mean(1,keepdim=True), d.amax(1,keepdim=True)),1)
                ga, gd = adapter(context).chunk(2, 1)
                # Bounded amplification AND suppression; not confidence masking.
                features.append(b.FFMs[stage-1](a*(1+.25*ga.tanh()), d*(1+.25*gd.tanh())))
        return {'logits': F.interpolate(self.decode_head(features), inputs.shape[-2:],
                                        mode='bilinear', align_corners=False)}
