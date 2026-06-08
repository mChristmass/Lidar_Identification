import math

import torch
import torch.nn as nn


class PixelGateNet(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=16, initial_c1_weight=0.9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        initial_c1_weight = min(max(initial_c1_weight, 1e-4), 1 - 1e-4)
        nn.init.constant_(final.bias, math.log(initial_c1_weight / (1 - initial_c1_weight)))

    def forward(self, x):
        return torch.sigmoid(self.net(x))[:, 0]
