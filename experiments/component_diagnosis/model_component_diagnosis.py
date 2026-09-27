from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


STAGES: Dict[str, tuple[str, ...]] = {
    "stage1": ("C0_C1L", "C1_HR_STD", "T1_NEW_EARLY"),
    "stage2": ("T2_STDENC_NEWDEC", "T3_NEWENC_STDDEC"),
    "stage3": ("T4_RES_PIXEL", "T5_CNX_MAXPOOL", "T6_CNX_LS1_PIXEL"),
    "stage4": (
        "T7_INDEP_IE_CONCAT",
        "T8_INDEP_IE_GATE",
        "T9_ADD_DEPTH_ZERO",
        "T10_ADD_SPATIAL",
    ),
    "stage5": ("C2_HR_STD_PM",),
}
VARIANTS = tuple(variant for stage in STAGES.values() for variant in stage)


@dataclass(frozen=True)
class VariantDescription:
    inputs: str
    encoder: str
    downsample: str
    decoder: str
    fusion: str


VARIANT_DESCRIPTIONS = {
    "C0_C1L": VariantDescription("I+E", "C1L DoubleConv H/16", "MaxPool", "C1L", "early"),
    "C1_HR_STD": VariantDescription("I+E", "DoubleConv H/8", "MaxPool", "strong", "early"),
    "C2_HR_STD_PM": VariantDescription(
        "I+E",
        "DoubleConv H/8 (parameter-matched width)",
        "MaxPool",
        "strong",
        "early",
    ),
    "T1_NEW_EARLY": VariantDescription("I+E", "ConvNeXtLite H/8", "PixelUnshuffle", "light", "early"),
    "T2_STDENC_NEWDEC": VariantDescription("I+E", "DoubleConv H/8", "MaxPool", "light", "early"),
    "T3_NEWENC_STDDEC": VariantDescription("I+E", "ConvNeXtLite H/8", "PixelUnshuffle", "strong", "early"),
    "T4_RES_PIXEL": VariantDescription("I+E", "ResidualDoubleConv H/8", "PixelUnshuffle", "strong", "early"),
    "T5_CNX_MAXPOOL": VariantDescription("I+E", "ConvNeXtLite H/8", "MaxPool+1x1", "strong", "early"),
    "T6_CNX_LS1_PIXEL": VariantDescription(
        "I+E", "ConvNeXtLite(layer_scale=1) H/8", "PixelUnshuffle", "strong", "early"
    ),
    "T7_INDEP_IE_CONCAT": VariantDescription(
        "I+E", "independent DoubleConv H/8", "MaxPool", "strong", "concat"
    ),
    "T8_INDEP_IE_GATE": VariantDescription(
        "I+E", "independent DoubleConv H/8", "MaxPool", "strong", "channel gate"
    ),
    "T9_ADD_DEPTH_ZERO": VariantDescription(
        "I+D+E", "independent DoubleConv H/8", "MaxPool", "strong", "channel gate + zero-depth"
    ),
    "T10_ADD_SPATIAL": VariantDescription(
        "I+D+E", "independent DoubleConv H/8", "MaxPool", "strong", "channel+spatial gate"
    ),
}


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ClassicDoubleConv(nn.Module):
    """Byte-for-byte layer choices of models/model_stage1.py::DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDoubleConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ConvNeXtLiteBlock(nn.Module):
    def __init__(self, channels: int, layer_scale_init: float = 1e-3):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, channels * 4, 1)
        self.project = nn.Conv2d(channels * 4, channels, 1)
        self.activation = nn.GELU()
        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(layer_scale_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.project(self.activation(self.expand(x)))
        return residual + self.layer_scale * x


class PixelDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.project = nn.Conv2d(in_channels * 4, out_channels, 1, bias=False)
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.project(self.unshuffle(x)))


class MaxProjectDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.project = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.project(self.pool(x)))


class StandardEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: Sequence[int]):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DoubleConv(in_channels if level == 0 else channels[level - 1], channel)
                for level, channel in enumerate(channels)
            ]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = []
        for level, block in enumerate(self.blocks):
            if level:
                x = self.pool(x)
            x = block(x)
            outputs.append(x)
        return tuple(outputs)


class ModernEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        *,
        downsample: str,
        block: str,
        layer_scale_init: float = 1e-3,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]) if block == "residual" else LayerNorm2d(channels[0]),
            nn.ReLU(inplace=False) if block == "residual" else nn.Identity(),
        )
        depths = (1, 1, 2, 2)
        if block == "residual":
            self.stages = nn.ModuleList(
                nn.Sequential(*(ResidualDoubleConv(channel) for _ in range(depth)))
                for channel, depth in zip(channels, depths)
            )
        else:
            self.stages = nn.ModuleList(
                nn.Sequential(
                    *(ConvNeXtLiteBlock(channel, layer_scale_init) for _ in range(depth))
                )
                for channel, depth in zip(channels, depths)
            )
        downsample_class = PixelDownsample if downsample == "pixel" else MaxProjectDownsample
        self.downsamples = nn.ModuleList(
            downsample_class(channels[level], channels[level + 1])
            for level in range(len(channels) - 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = []
        x = self.stem(x)
        for level, stage in enumerate(self.stages):
            x = stage(x)
            outputs.append(x)
            if level < len(self.downsamples):
                x = self.downsamples[level](x)
        return tuple(outputs)


class LightDecoderBlock(nn.Module):
    def __init__(self, high_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(high_channels + skip_channels, out_channels, 1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )
        self.refine = ConvNeXtLiteBlock(out_channels)

    def forward(self, high: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        high = F.interpolate(high, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(self.project(torch.cat([high, skip], dim=1)))


class StrongDecoderBlock(nn.Module):
    def __init__(self, high_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.high_projection = nn.Conv2d(high_channels, out_channels, 1, bias=False)
        self.fuse = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, high: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        high = F.interpolate(high, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.high_projection(high), skip], dim=1))


class PyramidDecoder(nn.Module):
    def __init__(self, channels: Sequence[int], decoder: str, num_classes: int):
        super().__init__()
        block = LightDecoderBlock if decoder == "light" else StrongDecoderBlock
        self.blocks = nn.ModuleList(
            block(channels[level + 1], channels[level], channels[level])
            for level in reversed(range(len(channels) - 1))
        )
        self.head = nn.Conv2d(channels[0], num_classes, 1)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        for block, skip in zip(self.blocks, reversed(features[:-1])):
            x = block(x, skip)
        return self.head(x)


class ClassicC1L(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, num_classes: int):
        super().__init__()
        channels = [base_channels * (2**level) for level in range(5)]
        self.encoders = nn.ModuleList(
            ClassicDoubleConv(
                in_channels if level == 0 else channels[level - 1], channel
            )
            for level, channel in enumerate(channels)
        )
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(channels[level], channels[level - 1], 2, stride=2)
            for level in reversed(range(1, len(channels)))
        )
        self.decoders = nn.ModuleList(
            ClassicDoubleConv(channels[level - 1] * 2, channels[level - 1])
            for level in reversed(range(1, len(channels)))
        )
        self.head = nn.Conv2d(channels[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for level, encoder in enumerate(self.encoders):
            if level:
                x = self.pool(x)
            x = encoder(x)
            features.append(x)
        x = features[-1]
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(features[:-1])):
            x = decoder(torch.cat([up(x), skip], dim=1))
        return self.head(x)


class ChannelGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(8, channels // 4)
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, main: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        descriptors = torch.cat(
            [F.adaptive_avg_pool2d(main, 1), F.adaptive_avg_pool2d(auxiliary, 1)], dim=1
        )
        return torch.sigmoid(self.net(descriptors))


class SpatialGate(nn.Module):
    def __init__(self, use_validity: bool):
        super().__init__()
        self.use_validity = use_validity
        self.net = nn.Conv2d(5 if use_validity else 4, 1, 7, padding=3)

    def forward(
        self,
        main: torch.Tensor,
        auxiliary: torch.Tensor,
        validity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        descriptors = [
            main.mean(1, keepdim=True),
            main.amax(1, keepdim=True),
            auxiliary.mean(1, keepdim=True),
            auxiliary.amax(1, keepdim=True),
        ]
        if self.use_validity:
            if validity is None:
                raise ValueError("Depth spatial gate requires validity")
            descriptors.append(F.interpolate(validity, main.shape[-2:], mode="nearest"))
        return torch.sigmoid(self.net(torch.cat(descriptors, dim=1)))


class IndependentFusion(nn.Module):
    def __init__(self, channels: int, mode: str, include_depth: bool):
        super().__init__()
        self.mode = mode
        self.include_depth = include_depth
        if mode == "concat":
            self.concat = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=False),
            )
        else:
            self.edge_channel = ChannelGate(channels)
            self.depth_channel = ChannelGate(channels) if include_depth else None
            self.edge_spatial = SpatialGate(False) if mode == "spatial" else None
            self.depth_spatial = SpatialGate(True) if mode == "spatial" and include_depth else None
            self.depth_alpha = nn.Parameter(torch.zeros(())) if include_depth else None
        self.refine = ResidualDoubleConv(channels)

    def forward(
        self,
        intensity: torch.Tensor,
        edge: torch.Tensor,
        depth: torch.Tensor | None,
        validity: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "concat":
            return self.refine(intensity + self.concat(torch.cat([intensity, edge], dim=1)))
        edge_gate = self.edge_channel(intensity, edge)
        if self.edge_spatial is not None:
            edge_gate = edge_gate * self.edge_spatial(intensity, edge)
        fused = intensity + edge_gate * edge
        if self.include_depth:
            if depth is None:
                raise ValueError("Depth feature is required")
            depth_gate = self.depth_channel(intensity, depth)
            if self.depth_spatial is not None:
                depth_gate = depth_gate * self.depth_spatial(intensity, depth, validity)
            fused = fused + self.depth_alpha * depth_gate * depth
        return self.refine(fused)


class ComponentDiagnosisNet(nn.Module):
    def __init__(
        self,
        variant: str,
        input_items: Sequence[str] = ("intensity", "depth", "local_depth_edge"),
        base_channels: int = 38,
        num_classes: int = 2,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
        self.variant = variant
        self.input_items = tuple(input_items)
        required = {"intensity", "depth", "local_depth_edge"}
        if not required.issubset(self.input_items):
            raise ValueError(f"input_items must contain {sorted(required)}")
        self.indices = {name: self.input_items.index(name) for name in required}
        channels = [base_channels * (2**level) for level in range(4)]

        if variant == "C0_C1L":
            self.classic = ClassicC1L(2, base_channels, num_classes)
            self.mode = "classic"
        elif variant.startswith(("T7_", "T8_", "T9_", "T10_")):
            self.mode = "independent"
            include_depth = variant in {"T9_ADD_DEPTH_ZERO", "T10_ADD_SPATIAL"}
            fusion_mode = {
                "T7_INDEP_IE_CONCAT": "concat",
                "T8_INDEP_IE_GATE": "channel",
                "T9_ADD_DEPTH_ZERO": "channel",
                "T10_ADD_SPATIAL": "spatial",
            }[variant]
            self.intensity_encoder = StandardEncoder(1, channels)
            self.edge_encoder = StandardEncoder(1, channels)
            self.depth_encoder = StandardEncoder(2, channels) if include_depth else None
            self.fusions = nn.ModuleList(
                IndependentFusion(channel, fusion_mode, include_depth) for channel in channels
            )
            self.decoder = PyramidDecoder(channels, "strong", num_classes)
        else:
            self.mode = "early"
            if variant in {"C1_HR_STD", "C2_HR_STD_PM", "T2_STDENC_NEWDEC"}:
                self.encoder = StandardEncoder(2, channels)
            elif variant == "T4_RES_PIXEL":
                self.encoder = ModernEncoder(
                    2, channels, downsample="pixel", block="residual"
                )
            else:
                downsample = "max" if variant == "T5_CNX_MAXPOOL" else "pixel"
                layer_scale = 1.0 if variant == "T6_CNX_LS1_PIXEL" else 1e-3
                self.encoder = ModernEncoder(
                    2,
                    channels,
                    downsample=downsample,
                    block="convnext",
                    layer_scale_init=layer_scale,
                )
            decoder = "light" if variant in {"T1_NEW_EARLY", "T2_STDENC_NEWDEC"} else "strong"
            self.decoder = PyramidDecoder(channels, decoder, num_classes)

        self.apply(self._initialize)
        self._reset_gate_initialization()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _reset_gate_initialization(self) -> None:
        for module in self.modules():
            if isinstance(module, ChannelGate):
                nn.init.zeros_(module.net[-1].weight)
                nn.init.constant_(module.net[-1].bias, -2.0)
            elif isinstance(module, SpatialGate):
                nn.init.zeros_(module.net.weight)
                nn.init.constant_(module.net.bias, -2.0)

    def _channel(self, x: torch.Tensor, name: str) -> torch.Tensor:
        index = self.indices[name]
        return x[:, index : index + 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intensity = self._channel(x, "intensity")
        depth = self._channel(x, "depth")
        edge = self._channel(x, "local_depth_edge")
        if self.mode == "classic":
            return self.classic(torch.cat([intensity, edge], dim=1))
        if self.mode == "early":
            return self.decoder(self.encoder(torch.cat([intensity, edge], dim=1)))

        validity = (depth > 0).to(depth.dtype)
        intensity_features = self.intensity_encoder(intensity)
        edge_features = self.edge_encoder(edge)
        depth_features = (
            self.depth_encoder(torch.cat([depth, validity], dim=1))
            if self.depth_encoder is not None
            else (None,) * len(intensity_features)
        )
        fused = [
            fusion(i, e, d, validity)
            for fusion, i, e, d in zip(
                self.fusions, intensity_features, edge_features, depth_features
            )
        ]
        return self.decoder(fused)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
