import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EdgeFusion(nn.Module):
    """Inject projected edge features into the intensity stream."""

    def __init__(self, intensity_channels, edge_channels, use_gate, inject_edge=True):
        super().__init__()
        self.use_gate = use_gate
        self.inject_edge = inject_edge
        if inject_edge:
            self.edge_projection = nn.Sequential(
                nn.Conv2d(edge_channels, intensity_channels, 1, bias=False),
                nn.BatchNorm2d(intensity_channels),
            )
        else:
            self.edge_projection = None

        if use_gate and inject_edge:
            hidden_channels = max(8, intensity_channels // 4)
            self.gate = nn.Sequential(
                nn.Conv2d(intensity_channels * 2, hidden_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, 1, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.gate[-2].weight)
            nn.init.constant_(self.gate[-2].bias, 0.0)
        else:
            self.gate = None

    def forward(self, intensity_features, edge_features):
        if not self.inject_edge:
            gate = torch.zeros_like(intensity_features[:, :1])
            return intensity_features, gate

        projected_edge = self.edge_projection(edge_features)
        if self.gate is None:
            gate = torch.ones_like(projected_edge[:, :1])
        else:
            gate = self.gate(torch.cat([intensity_features, projected_edge], dim=1))
        return intensity_features + gate * projected_edge, gate


class DualBranchGatedUNet(nn.Module):
    """
    Intensity is the main stream. Local depth edge has a smaller encoder and is
    injected through spatial gates at selected scales.
    """

    VALID_FUSION_MODES = {
        "no_gate",
        "bottleneck_gate",
        "multiscale_gate",
        "shallow_gate",
    }

    def __init__(
        self,
        num_classes=2,
        intensity_base_channels=32,
        edge_base_channels=16,
        fusion_mode="multiscale_gate",
    ):
        super().__init__()
        if fusion_mode not in self.VALID_FUSION_MODES:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

        self.fusion_mode = fusion_mode
        ic = [intensity_base_channels * (2**i) for i in range(5)]
        ec = [edge_base_channels * (2**i) for i in range(5)]
        self.pool = nn.MaxPool2d(2)

        self.intensity_encoders = nn.ModuleList(
            [
                DoubleConv(1, ic[0]),
                DoubleConv(ic[0], ic[1]),
                DoubleConv(ic[1], ic[2]),
                DoubleConv(ic[2], ic[3]),
                DoubleConv(ic[3], ic[4]),
            ]
        )
        self.edge_encoders = nn.ModuleList(
            [
                DoubleConv(1, ec[0]),
                DoubleConv(ec[0], ec[1]),
                DoubleConv(ec[1], ec[2]),
                DoubleConv(ec[2], ec[3]),
                DoubleConv(ec[3], ec[4]),
            ]
        )

        self.fusions = nn.ModuleList()
        for scale in range(5):
            use_gate = fusion_mode == "multiscale_gate" or (
                fusion_mode == "bottleneck_gate" and scale == 4
            ) or (
                fusion_mode == "shallow_gate" and scale < 3
            )
            inject_edge = fusion_mode != "shallow_gate" or scale < 3
            self.fusions.append(
                EdgeFusion(
                    ic[scale],
                    ec[scale],
                    use_gate=use_gate,
                    inject_edge=inject_edge,
                )
            )

        self.up4 = nn.ConvTranspose2d(ic[4], ic[3], 2, stride=2)
        self.dec4 = DoubleConv(ic[3] * 2, ic[3])
        self.up3 = nn.ConvTranspose2d(ic[3], ic[2], 2, stride=2)
        self.dec3 = DoubleConv(ic[2] * 2, ic[2])
        self.up2 = nn.ConvTranspose2d(ic[2], ic[1], 2, stride=2)
        self.dec2 = DoubleConv(ic[1] * 2, ic[1])
        self.up1 = nn.ConvTranspose2d(ic[1], ic[0], 2, stride=2)
        self.dec1 = DoubleConv(ic[0] * 2, ic[0])

        self.segmentation_head = nn.Conv2d(ic[0], num_classes, 1)
        self.boundary_head = nn.Conv2d(ic[0], 1, 1)

    @staticmethod
    def _resize_like(x, reference):
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x, return_aux=False):
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(f"Expected input [B, 2, H, W], got {tuple(x.shape)}")

        intensity = x[:, 0:1]
        edge = x[:, 1:2]
        fused_skips = []
        gates = []

        for scale, (intensity_encoder, edge_encoder, fusion) in enumerate(
            zip(self.intensity_encoders, self.edge_encoders, self.fusions)
        ):
            intensity = intensity_encoder(intensity)
            edge = edge_encoder(edge)
            fused, gate = fusion(intensity, edge)
            fused_skips.append(fused)
            gates.append(gate)
            if scale < 4:
                intensity = self.pool(fused)
                edge = self.pool(edge)

        d4 = self.dec4(torch.cat([self._resize_like(self.up4(fused_skips[4]), fused_skips[3]), fused_skips[3]], dim=1))
        d3 = self.dec3(torch.cat([self._resize_like(self.up3(d4), fused_skips[2]), fused_skips[2]], dim=1))
        d2 = self.dec2(torch.cat([self._resize_like(self.up2(d3), fused_skips[1]), fused_skips[1]], dim=1))
        d1 = self.dec1(torch.cat([self._resize_like(self.up1(d2), fused_skips[0]), fused_skips[0]], dim=1))

        logits = self.segmentation_head(d1)
        if not return_aux:
            return logits
        return {
            "logits": logits,
            "boundary_logits": self.boundary_head(d1).squeeze(1),
            "gates": gates,
        }
