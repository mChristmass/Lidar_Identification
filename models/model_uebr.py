import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_stage1 import DoubleConv


class C1LFeatureUNet(nn.Module):
    def __init__(self, base_channels=38, num_classes=2):
        super().__init__()
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
        self.output_channels = channels[0]

    @staticmethod
    def _resize_like(x, reference):
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self._resize_like(self.up4(bottleneck), e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._resize_like(self.up3(d4), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._resize_like(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._resize_like(self.up1(d2), e1), e1], dim=1))
        return self.segmentation_head(d1), d1


class ShallowEdgeEncoder(nn.Module):
    def __init__(self, out_channels=16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, edge):
        return self.block(edge)


class UncertaintyEdgeBoundaryRefinementNet(nn.Module):
    VALID_MODES = {"boundary_only", "full_residual", "uncertainty_residual"}

    def __init__(
        self,
        base_channels=38,
        edge_feature_channels=16,
        refinement_channels=32,
        refinement_mode="uncertainty_residual",
        use_explicit_edge=True,
        refinement_scale=1.0,
    ):
        super().__init__()
        if refinement_mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported refinement mode: {refinement_mode}")
        self.refinement_mode = refinement_mode
        self.use_explicit_edge = bool(use_explicit_edge)
        self.refinement_scale = float(refinement_scale)

        self.base = C1LFeatureUNet(base_channels=base_channels, num_classes=2)
        if self.refinement_mode == "boundary_only":
            self.edge_encoder = None
            self.refinement = None
        else:
            self.edge_encoder = ShallowEdgeEncoder(edge_feature_channels)
            refinement_inputs = self.base.output_channels + edge_feature_channels + 2
            self.refinement = nn.Sequential(
                nn.Conv2d(refinement_inputs, refinement_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(refinement_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(refinement_channels, refinement_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(refinement_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(refinement_channels, 2, 1),
            )
            nn.init.zeros_(self.refinement[-1].weight)
            nn.init.zeros_(self.refinement[-1].bias)
        self.boundary_head = nn.Conv2d(self.base.output_channels, 1, 1)

    def forward(self, x, return_aux=False):
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"Expected input [B, 4, H, W], got {tuple(x.shape)}")
        intensity = x[:, 0:1]
        edge = x[:, 3:4]
        base_logits, decoder_features = self.base(torch.cat([intensity, edge], dim=1))

        foreground_probability = torch.softmax(base_logits, dim=1)[:, 1:2]
        uncertainty = (4.0 * foreground_probability * (1.0 - foreground_probability)).detach()
        if self.refinement_mode == "boundary_only":
            delta_logits = torch.zeros_like(base_logits)
            correction_mask = torch.zeros_like(uncertainty)
        else:
            edge_input = edge if self.use_explicit_edge else torch.zeros_like(edge)
            edge_features = self.edge_encoder(edge_input)
            refinement_input = torch.cat(
                [decoder_features, edge_features, foreground_probability, uncertainty],
                dim=1,
            )
            delta_logits = self.refinement(refinement_input)
            correction_mask = (
                torch.ones_like(uncertainty)
                if self.refinement_mode == "full_residual"
                else uncertainty
            )
        final_logits = base_logits + self.refinement_scale * correction_mask * delta_logits

        if not return_aux:
            return final_logits
        return {
            "logits": final_logits,
            "base_logits": base_logits,
            "boundary_logits": self.boundary_head(decoder_features).squeeze(1),
            "uncertainty": uncertainty.squeeze(1),
            "delta_logits": delta_logits,
            "correction_mask": correction_mask.squeeze(1),
        }
