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


class ReliabilityEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=12):
        super().__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, base_channels),
            nn.Conv2d(base_channels, base_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x):
        return self.head(self.encoder(x))


class ReliabilityGuidedEdgeUNet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        base_channels=38,
        reliability_base_channels=12,
        edge_keep_alpha=0.25,
        use_raw_depth=True,
        modulation_mode="suppress",
        gate_beta=0.25,
    ):
        super().__init__()
        self.edge_keep_alpha = float(edge_keep_alpha)
        self.use_raw_depth = bool(use_raw_depth)
        self.modulation_mode = modulation_mode
        self.gate_beta = float(gate_beta)
        if self.modulation_mode not in {"suppress", "residual"}:
            raise ValueError(f"Unsupported modulation mode: {self.modulation_mode}")
        reliability_channels = 3 if self.use_raw_depth else 2
        self.reliability = ReliabilityEncoder(
            in_channels=reliability_channels,
            base_channels=reliability_base_channels,
        )

        channels = [base_channels * (2**level) for level in range(5)]
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(2, channels[0])
        self.enc2 = DoubleConv(channels[0], channels[1])
        self.enc3 = DoubleConv(channels[1], channels[2])
        self.enc4 = DoubleConv(channels[2], channels[3])
        self.bottleneck = DoubleConv(channels[3], channels[4])

        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2)
        self.dec4 = DoubleConv(channels[3] * 2, channels[3])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3 = DoubleConv(channels[2] * 2, channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2 = DoubleConv(channels[1] * 2, channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.dec1 = DoubleConv(channels[0] * 2, channels[0])

        self.segmentation_head = nn.Conv2d(channels[0], num_classes, 1)
        self.boundary_head = nn.Conv2d(channels[0], 1, 1)

    @staticmethod
    def _resize_like(x, reference):
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x, return_aux=False):
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"Expected input [B, 4, H, W], got {tuple(x.shape)}")

        intensity = x[:, 0:1]
        depth = x[:, 1:2]
        valid = x[:, 2:3]
        edge = x[:, 3:4]
        if self.use_raw_depth:
            reliability_input = torch.cat([depth, valid, edge], dim=1)
        else:
            reliability_input = torch.cat([valid, edge], dim=1)

        reliability_logits = self.reliability(reliability_input)
        reliability = torch.sigmoid(reliability_logits)
        if self.modulation_mode == "residual":
            edge_weight = 1.0 + self.gate_beta * (reliability - 0.5)
        else:
            edge_weight = self.edge_keep_alpha + (1.0 - self.edge_keep_alpha) * reliability
        filtered_edge = edge * edge_weight

        e1 = self.enc1(torch.cat([intensity, filtered_edge], dim=1))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self._resize_like(self.up4(bottleneck), e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._resize_like(self.up3(d4), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._resize_like(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._resize_like(self.up1(d2), e1), e1], dim=1))
        logits = self.segmentation_head(d1)

        if not return_aux:
            return logits
        return {
            "logits": logits,
            "boundary_logits": self.boundary_head(d1).squeeze(1),
            "reliability_logits": reliability_logits.squeeze(1),
            "reliability": reliability.squeeze(1),
            "filtered_edge": filtered_edge.squeeze(1),
        }
