"""Raw-observation control for S, and segmentation-supervised intensity correction.

No clean target, teacher, restoration checkpoint or inferred noise label is used.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .model import CMXPaper
from .structure_model import StructureCMX, conv_block

CORRECTION_WEIGHT = .01


class TaskControlCMX(StructureCMX):
    def __init__(self, method, variant='CMX_B2', num_classes=40, image_size=640, decoder_dim=256):
        if method not in ('B0', 'S_RAW', 'S', 'R_TASK'):
            raise ValueError(method)
        super().__init__('S' if method in ('S', 'S_RAW') else 'B0',
                         variant, num_classes, image_size, decoder_dim)
        self.experiment = method
        if method == 'S_RAW':
            # Same 32-channel pyramid/exchanges and injection sites; only first input is 3 vs 8.
            # Replace after constructing S so all common initial parameters remain identical.
            self.adapters['pyramid'][0][0] = conv_block(3, 32, 2)
        elif method == 'R_TASK':
            # Same 2913-parameter architecture as historical D1, newly initialized per domain.
            self.adapters = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GELU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 3, padding=1))
            nn.init.zeros_(self.adapters[-1].weight)
            nn.init.zeros_(self.adapters[-1].bias)

    def forward(self, inputs):
        self.audit_rows = []
        if not self.adapters_enabled or self.experiment == 'B0':
            return CMXPaper.forward(self, inputs)
        if self.experiment == 'S':
            return super().forward(inputs)
        if self.experiment == 'R_TASK':
            delta = self.adapters(inputs)
            intensity = inputs[:, :1]
            corrected = (intensity + delta).clamp(0, 1)
            out = CMXPaper.forward(self, torch.cat((corrected, inputs[:, 1:]), 1))
            # Penalize RAW correction, not clipped correction: clipping must not hide runaway values.
            out['correction_penalty'] = delta.float().square().mean()
            if self.audit_enabled:
                applied = (corrected-intensity).detach().float()
                self.audit_rows = [dict(stage='input', raw_delta_rms=float(delta.detach().float().square().mean().sqrt()),
                    applied_mae=float(applied.abs().mean()), applied_max=float(applied.abs().max()),
                    clipped_fraction=float(((intensity+delta<0)|(intensity+delta>1)).float().mean()))]
            return out
        # S_RAW: no explicit local support/residual/std descriptors; only observed I,D,M.
        structure = inputs[:, :3]
        a, d, b, features = inputs[:, :1], inputs[:, 1:2], self.backbone, []
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
            structure = self.adapters['pyramid'][stage-1](structure)
            structure = F.adaptive_avg_pool2d(structure, (h, w))
            (da, dd), weights = self.adapters['exchange'][stage-1](a, d, structure)
            self.record(stage, a, d, da, dd, weights)
            a, d = b.FRMs[stage-1](a+da, d+dd)
            features.append(b.FFMs[stage-1](a, d))
        return {'logits': F.interpolate(self.decode_head(features), inputs.shape[-2:],
                                        mode='bilinear', align_corners=False)}


def task_loss(prediction, labels):
    """Semantic labels only. R_TASK's L2 is a regularizer, NOT a clean-image target."""
    ce = F.cross_entropy(prediction['logits'], labels, ignore_index=255)
    penalty = prediction.get('correction_penalty', ce.new_zeros(()))
    return ce + CORRECTION_WEIGHT*penalty, ce, penalty
