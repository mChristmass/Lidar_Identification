"""Regional CA and detached semantic-evidence gates for the original CMX FRM.

Auxiliary heads are probes, not independent unimodal networks. Only stage 0
has never received cross-modal features. Entropy is a confidence proxy.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('B', 'P', 'G', 'H', 'PG')


def evidence(a, b):
    p, q = a.detach().float().softmax(1), b.detach().float().softmax(1)
    m = (p + q) / 2
    entropy = lambda x: -(x * x.clamp_min(1e-8).log()).sum(1, keepdim=True)
    js = (entropy(m) - (entropy(p) + entropy(q)) / 2).clamp_min(0)
    return (1-js/math.log(2)).clamp(0, 1), (1-entropy(p)/math.log(p.shape[1])).clamp(0, 1), (1-entropy(q)/math.log(q.shape[1])).clamp(0, 1)


class RegionalFRM(nn.Module):
    def __init__(self, original, classes, arm):
        super().__init__()
        self.original, self.arm = original, arm
        dim = original.channel_weights.dim
        self.probes = nn.ModuleList([nn.Conv2d(dim, classes, 1) for _ in range(2)])
        self.register_buffer('threshold', torch.tensor(.9))
        # The same existing CA MLP operates on a 4x4 pooled regional grid.
        self.alpha = nn.Parameter(torch.tensor(0.), requires_grad=arm in ('P', 'PG'))
        self.capture = False
        self.maps = {}
        self.logits = None

    def forward(self, x, y):
        base = self.original
        cw = base.channel_weights(x, y)
        sw = base.spatial_weights(x, y)
        local = cw
        grid = cw
        if self.arm in ('P', 'PG') or self.capture:
            z = torch.cat((x, y), 1)
            gh, gw = min(4, x.shape[-2]), min(4, x.shape[-1])
            stats = torch.cat((F.adaptive_avg_pool2d(z, (gh, gw)),
                               F.adaptive_max_pool2d(z, (gh, gw))), 1)
            weights = base.channel_weights.mlp(stats.permute(0, 2, 3, 1))
            weights = weights.permute(0, 3, 1, 2)
            grid = weights.reshape(x.shape[0], 2, x.shape[1], gh, gw).transpose(0, 1)
            weights = F.interpolate(weights, x.shape[-2:], mode='bilinear', align_corners=False)
            local = weights.reshape(x.shape[0], 2, x.shape[1], *x.shape[-2:]).transpose(0, 1)
        # A nonzero local contribution makes P/PG an actual regional-CA test.
        # v1 initialized this blend at zero and it stayed near zero in all seeds.
        mix = .5 + .5*self.alpha.clamp(0, 1)
        effective = cw + mix * (local-cw) if self.arm in ('P', 'PG') else cw
        self.logits = [head(z.detach()) for head, z in zip(self.probes, (x, y))]
        c, ri, rd = evidence(*self.logits)
        gi, gd = torch.ones_like(c), torch.ones_like(c)
        if self.arm == 'G' or self.arm == 'PG':
            gi = rd * (c + (1-c)*(1-ri))  # D -> I
            gd = ri * (c + (1-c)*(1-rd))  # I -> D
        elif self.arm == 'H':
            gi = gd = (c >= self.threshold).to(c.dtype)
        di = (base.lambda_c*effective[1] + base.lambda_s*sw[1])*y
        dd = (base.lambda_c*effective[0] + base.lambda_s*sw[0])*x
        if self.capture:
            self.maps = dict(consistency=c.detach(), confidence_i=ri, confidence_d=rd,
                gate_d_to_i=gi, gate_i_to_d=gd, ca_global=cw.detach(),
                ca_local_grid=grid.detach(),
                ca_local=local.detach(), ca_effective=effective.detach(),
                spatial_weights=sw.detach(),
                delta_i_relative=(gi*di).detach().square().mean(1, keepdim=True).sqrt() /
                    x.detach().square().mean(1, keepdim=True).sqrt().clamp_min(1e-6),
                delta_d_relative=(gd*dd).detach().square().mean(1, keepdim=True).sqrt() /
                    y.detach().square().mean(1, keepdim=True).sqrt().clamp_min(1e-6))
        return x + gi.to(x.dtype)*di, y + gd.to(y.dtype)*dd


def attach(model, classes, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    model.backbone.FRMs = nn.ModuleList([RegionalFRM(m, classes, arm) for m in model.backbone.FRMs])
    return model


def probe_loss(model, target):
    losses = []
    for module in model.backbone.FRMs:
        for logits in module.logits:
            logits = F.interpolate(logits, target.shape[-2:], mode='bilinear', align_corners=False)
            losses.append(F.cross_entropy(logits, target.long(), ignore_index=255))
    return torch.stack(losses).mean()
