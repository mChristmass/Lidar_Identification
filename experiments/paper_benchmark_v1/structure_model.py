"""Observation structure/aggregation hypotheses, not calibrated uncertainty."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import CMXPaper, BACKBONES
from .observation_model import ObservationCMX, observation_descriptors


def conv_block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                         nn.GroupNorm(4, cout), nn.GELU())


def zero_output(cin, cout):
    layer = nn.Conv2d(cin, cout, 1)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def neighbors(x):
    """B,C,9,H,W with replicate boundaries, no fictitious padded observations."""
    b, c, h, w = x.shape
    return F.unfold(F.pad(x, (1, 1, 1, 1), mode='replicate'), 3).reshape(b, c, 9, h, w)


class StructureExchange(nn.Module):
    def __init__(self, channels, width=32):
        super().__init__()
        self.appearance = nn.Conv2d(channels, width, 1)
        self.depth = nn.Conv2d(channels, width, 1)
        self.mix = conv_block(width * 3, width)
        self.out = zero_output(width, channels * 2)

    def forward(self, a, d, structure):
        context = torch.cat((self.appearance(a), self.depth(d), structure), 1)
        return self.out(self.mix(context)).chunk(2, 1), None


class SupportAggregation(nn.Module):
    """Two queries gather 3x3 appearance/depth tokens and update BOTH streams."""
    def __init__(self, channels, width=32):
        super().__init__()
        self.appearance = nn.Conv2d(channels, width, 1)
        self.depth = nn.Conv2d(channels, width, 1)
        self.context = conv_block(width * 2 + 8, width)
        self.query = nn.Conv2d(width, 32, 1)
        self.keys = nn.ModuleList([nn.Conv2d(width + 8, 16, 1) for _ in range(2)])
        self.out = zero_output(width * 2, channels * 2)

    def forward(self, a, d, desc):
        av, dv = self.appearance(a), self.depth(d)
        queries = self.query(self.context(torch.cat((av, dv, desc), 1))).chunk(2, 1)
        keys = torch.cat([neighbors(layer(torch.cat((v, desc), 1)))
                          for layer, v in zip(self.keys, (av, dv))], 2)
        values = torch.cat((neighbors(av), neighbors(dv)), 2)
        support = neighbors(desc[:, :1])[:, 0].float().clamp(0, 1)
        prior = torch.cat((torch.ones_like(support), support), 1)
        weights, outputs = [], []
        for query in queries:
            with torch.autocast(device_type=a.device.type, enabled=False):
                logits = (query.float().unsqueeze(2) * keys.float()).sum(1) / 4.
                logits = logits + prior.clamp_min(1e-6).log()
                weight = logits.masked_fill(prior == 0, float('-inf')).softmax(1)
            weights.append(weight)
            outputs.append((values * weight[:, None].to(values.dtype)).sum(2))
        return self.out(torch.cat(outputs, 1)).chunk(2, 1), torch.stack(weights, 1)


class StructureCMX(ObservationCMX):
    def __init__(self, method, variant='CMX_B2', num_classes=40, image_size=640, decoder_dim=256):
        if method not in ('B0', 'L', 'S', 'A'):
            raise ValueError(method)
        super().__init__('L' if method == 'L' else 'B0', variant, num_classes, image_size, decoder_dim)
        self.method = method
        self.adapters_enabled = True
        self.audit_enabled = False
        self.audit_rows = []
        if method == 'S':
            self.adapters = nn.ModuleDict({
                'pyramid': nn.ModuleList([
                    nn.Sequential(conv_block(8, 32, 2), conv_block(32, 32, 2)),
                    *[conv_block(32, 32, 2) for _ in range(3)]]),
                'exchange': nn.ModuleList([StructureExchange(c) for c in BACKBONES[variant][1]])})
        elif method == 'A':
            self.adapters = nn.ModuleList([SupportAggregation(c) for c in BACKBONES[variant][1]])

    def record(self, stage, a, d, da, dd, weights=None):
        if not self.audit_enabled:
            return
        def rms(x):
            return x.detach().float().square().mean().sqrt().item()
        row = dict(stage=stage, appearance_residual_rms=rms(da), depth_residual_rms=rms(dd),
                   appearance_relative_rms=rms(da)/max(rms(a), 1e-8),
                   depth_relative_rms=rms(dd)/max(rms(d), 1e-8))
        if weights is not None:
            w = weights.detach().float()
            row['depth_mass_by_query'] = w[:, :, 9:].sum(2).mean((0, 2, 3)).tolist()
            row['routing_entropy'] = (-(w * w.clamp_min(1e-8).log()).sum(2)).mean().item()
        self.audit_rows.append(row)

    def forward(self, inputs):
        self.audit_rows = []
        if not self.adapters_enabled or self.method == 'B0':
            return CMXPaper.forward(self, inputs)
        if self.method == 'L':
            features, changed = self.backbone(inputs[:, :1], inputs[:, 1:2]), []
            for stage, (f, adapter) in enumerate(zip(features, self.adapters), 1):
                delta = adapter(f)
                self.record(stage, f, f, delta, delta)
                changed.append(f + delta)
            features = changed
        else:
            desc = observation_descriptors(inputs)
            structure = desc
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
                if self.method == 'S':
                    structure = self.adapters['pyramid'][stage-1](structure)
                    structure = F.adaptive_avg_pool2d(structure, (h, w))
                    (da, dd), weights = self.adapters['exchange'][stage-1](a, d, structure)
                else:
                    (da, dd), weights = self.adapters[stage-1](a, d, F.adaptive_avg_pool2d(desc, (h, w)))
                self.record(stage, a, d, da, dd, weights)
                a, d = b.FRMs[stage-1](a + da, d + dd)
                features.append(b.FFMs[stage-1](a, d))
                # Updated streams ALSO feed the next encoder, unlike observation C.
        return {'logits': F.interpolate(self.decode_head(features), inputs.shape[-2:],
                                        mode='bilinear', align_corners=False)}
