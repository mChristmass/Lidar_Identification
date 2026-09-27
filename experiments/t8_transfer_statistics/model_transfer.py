from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from experiments.component_diagnosis.model_component_diagnosis import (
    ChannelGate, IndependentFusion, PyramidDecoder, ResidualDoubleConv, StandardEncoder,
)


@dataclass(frozen=True)
class Description:
    coarse_head: bool = False
    guidance: bool = False
    dem_fusion: str = "none"
    context: str = "none"


VARIANTS = {
    "T0_R1_REPRO": Description(),
    "T1_COARSE_AUX": Description(coarse_head=True),
    "T2_COARSE_GUIDANCE": Description(coarse_head=True, guidance=True),
    "T3_DEM_CHANNEL_ONLY": Description(dem_fusion="channel"),
    "T3_DEM_FUSION": Description(dem_fusion="full"),  # compatibility: full DEM
    "T4_CONTEXT_STD": Description(context="standard"),
    "T4_MULTISCALE_CONTEXT": Description(context="light"),  # compatibility: light context
    "T5_GUIDANCE_DEM_CHANNEL": Description(coarse_head=True, guidance=True, dem_fusion="channel"),
    "T5_GUIDANCE_DEM": Description(coarse_head=True, guidance=True, dem_fusion="full"),
    "T6_T1_CONTEXT_STD": Description(coarse_head=True, context="standard"),
    "T6_T2_CONTEXT_STD": Description(coarse_head=True, guidance=True, context="standard"),
    "T6_T3C_CONTEXT_STD": Description(dem_fusion="channel", context="standard"),
    "T6_T5C_CONTEXT_STD": Description(coarse_head=True, guidance=True, dem_fusion="channel", context="standard"),
    "T6_T1_CONTEXT": Description(coarse_head=True, context="light"),
    "T6_T2_CONTEXT": Description(coarse_head=True, guidance=True, context="light"),
    "T6_T3_CONTEXT": Description(dem_fusion="full", context="light"),
    "T6_T5_CONTEXT": Description(coarse_head=True, guidance=True, dem_fusion="full", context="light"),
}
DEFAULT_VARIANTS = (
    "T0_R1_REPRO", "T1_COARSE_AUX", "T2_COARSE_GUIDANCE",
    "T3_DEM_CHANNEL_ONLY", "T3_DEM_FUSION", "T4_CONTEXT_STD", "T4_MULTISCALE_CONTEXT",
)


class DEMFusion(nn.Module):
    """BBSNet-style auxiliary channel+spatial selection followed by T8 refinement."""

    def __init__(self, channels: int, spatial: bool = True):
        super().__init__()
        hidden = max(1, channels // 16)
        self.channel = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=False), nn.Conv2d(hidden, channels, 1, bias=False), nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 1, 7, padding=3, bias=False), nn.Sigmoid(),
        ) if spatial else None
        self.refine = ResidualDoubleConv(channels)

    def forward(self, main, auxiliary):
        selected = auxiliary * self.channel(auxiliary)
        if self.spatial is not None:
            selected = selected * self.spatial(selected.amax(1, keepdim=True))
        return self.refine(main + selected)


class CoarseHead(nn.Module):
    """The same high-level head is shared by T1/T2/T5/T6 variants."""

    def __init__(self, channels, num_classes=2):
        super().__init__()
        width = channels[1]
        self.deep = nn.Conv2d(channels[3], width, 1, bias=False)
        self.mid = nn.Conv2d(channels[2], width, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=False),
            nn.Conv2d(width, num_classes, 1),
        )

    def forward(self, features, output_size):
        target = features[2].shape[-2:]
        deep = F.interpolate(self.deep(features[3]), target, mode="bilinear", align_corners=False)
        logits = self.fuse(torch.cat((self.mid(features[2]), deep), 1))
        return F.interpolate(logits, output_size, mode="bilinear", align_corners=False)


class LightMultiScaleContext(nn.Module):
    """Three single-path depthwise dilation views with a zero-start residual."""

    def __init__(self, channels):
        super().__init__()
        width = max(8, channels // 4)
        self.reduce = nn.Sequential(nn.Conv2d(channels, width, 1, bias=False),
                                    nn.BatchNorm2d(width), nn.ReLU(inplace=False))
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=d, dilation=d,
                                    groups=width, bias=False),
                          nn.BatchNorm2d(width), nn.ReLU(inplace=False))
            for d in (1, 2, 3)
        )
        self.project = nn.Sequential(nn.Conv2d(width * 3, channels, 1, bias=False),
                                     nn.BatchNorm2d(channels))
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, feature):
        reduced = self.reduce(feature)
        return feature + self.gamma * self.project(torch.cat([b(reduced) for b in self.branches], 1))


class StandardMultiScaleContext(nn.Module):
    """Standard-convolution control for the same dilation layout and residual interface."""

    def __init__(self, channels):
        super().__init__()
        width = max(8, channels // 4)
        self.reduce = nn.Sequential(nn.Conv2d(channels, width, 1, bias=False),
                                    nn.BatchNorm2d(width), nn.ReLU(inplace=False))
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=d, dilation=d, bias=False),
                          nn.BatchNorm2d(width), nn.ReLU(inplace=False))
            for d in (1, 2, 3)
        )
        self.project = nn.Sequential(nn.Conv2d(width * 3, channels, 1, bias=False),
                                     nn.BatchNorm2d(channels))
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, feature):
        reduced = self.reduce(feature)
        return feature + self.gamma * self.project(torch.cat([b(reduced) for b in self.branches], 1))


class T8TransferNet(nn.Module):
    def __init__(self, variant, base_channels=38, num_classes=2):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.description = variant, VARIANTS[variant]
        channels = [base_channels * 2**i for i in range(4)]
        self.channels = channels

        # Build and initialize the complete R1 graph first. This preserves common
        # initialization across variants under the same fold seed.
        self.intensity_encoder = StandardEncoder(1, channels)
        self.depth_encoder = StandardEncoder(1, channels)
        self.fusions = nn.ModuleList(
            IndependentFusion(c, mode="channel", include_depth=False) for c in channels
        )
        self.decoder = PyramidDecoder(channels, "strong", num_classes)
        self.apply(self._initialize)
        for module in self.modules():
            if isinstance(module, ChannelGate):
                nn.init.zeros_(module.net[-1].weight)
                nn.init.constant_(module.net[-1].bias, -2.0)

        if self.description.dem_fusion != "none":
            use_spatial = self.description.dem_fusion == "full"
            self.fusions = nn.ModuleList(DEMFusion(c, spatial=use_spatial) for c in channels)
            self.fusions.apply(self._initialize)
        self.coarse_head = CoarseHead(channels, num_classes) if self.description.coarse_head else None
        if self.coarse_head is not None:
            self.coarse_head.apply(self._initialize)
        self.contexts = nn.ModuleDict()
        if self.description.context != "none":
            # H/4 carries shape detail; H/8 supplies larger context.
            context_class = StandardMultiScaleContext if self.description.context == "standard" else LightMultiScaleContext
            self.contexts["2"] = context_class(channels[2])
            self.contexts["3"] = context_class(channels[3])
            self.contexts.apply(self._initialize)
            for module in self.contexts.values():
                nn.init.zeros_(module.gamma)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs):
        intensity, depth = inputs[:, :1], inputs[:, 1:2]
        main, auxiliary = self.intensity_encoder(intensity), self.depth_encoder(depth)
        empty = torch.empty(0, device=inputs.device)
        if self.description.dem_fusion != "none":
            fused = [fusion(a, b) for fusion, a, b in zip(self.fusions, main, auxiliary)]
        else:
            fused = [fusion(a, b, None, empty) for fusion, a, b in zip(self.fusions, main, auxiliary)]
        if self.description.context != "none":
            fused[2] = self.contexts["2"](fused[2])
            fused[3] = self.contexts["3"](fused[3])
        output = {}
        if self.coarse_head is not None:
            coarse = self.coarse_head(fused, inputs.shape[-2:])
            output["coarse_logits"] = coarse
            if self.description.guidance:
                saliency = torch.softmax(coarse, 1)[:, 1:2]
                # Fixed BBSNet form, with no learnable coefficient.
                for level in (0, 1, 2):
                    attention = F.interpolate(saliency, fused[level].shape[-2:],
                                              mode="bilinear", align_corners=False)
                    fused[level] = fused[level] + fused[level] * attention
        output["logits"] = self.decoder(fused)
        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def diagnostics(model):
    return {f"context_gamma_{key}": float(value.gamma.detach().cpu())
            for key, value in model.contexts.items()}
