import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
import configs.stage2.config_stage2 as conf


# =========================================
# 基础模块
# =========================================
class DoubleConv(nn.Module):
    """
    两层卷积：Conv-BN-ReLU * 2
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# =========================================
# UNet 主体
# =========================================
class Stage2UNet(nn.Module):
    """
    输入:
        x: [B, C, 128, 128]

    输出:
        logits: [B, 2, 128, 128]
    """

    def __init__(self, in_channels=conf.INPUT_CHANNEL, num_classes=2, base_ch=32):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        # 输出层
        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.out_conv(d1)

        return logits




# 将encoder更换为resnet18
# import torch
# import torch.nn as nn
# import torchvision.models as models
#
# import os
# import sys
#
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(PROJECT_ROOT)
#
# import configs.stage2.config_stage2 as conf
#
#
# # =========================================
# # DoubleConv
# # =========================================
# class DoubleConv(nn.Module):
#
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# # =========================================
# # Stage2 ResUNet
# # =========================================
# class Stage2UNet(nn.Module):
#
#     def __init__(
#             self,
#             in_channels=conf.INPUT_CHANNEL,
#             num_classes=2
#     ):
#
#         super().__init__()
#
#         # =====================================
#         # ResNet18
#         # 不使用预训练
#         # =====================================
#
#         resnet = models.resnet18(weights=None)
#
#         # -------------------------------------
#         # 修改输入通道
#         # -------------------------------------
#
#         if in_channels != 3:
#
#             resnet.conv1 = nn.Conv2d(
#                 in_channels,
#                 64,
#                 kernel_size=7,
#                 stride=2,
#                 padding=3,
#                 bias=False
#             )
#
#         # =====================================
#         # Encoder
#         # =====================================
#
#         self.input_layer = nn.Sequential(
#             resnet.conv1,
#             resnet.bn1,
#             resnet.relu
#         )
#
#         self.maxpool = resnet.maxpool
#
#         self.encoder1 = resnet.layer1   # 64
#         self.encoder2 = resnet.layer2   # 128
#         self.encoder3 = resnet.layer3   # 256
#         self.encoder4 = resnet.layer4   # 512
#
#         # =====================================
#         # Bottleneck
#         # =====================================
#
#         self.bottleneck = DoubleConv(512, 1024)
#
#         # =====================================
#         # Decoder
#         # =====================================
#
#         self.up4 = nn.ConvTranspose2d(
#             1024,
#             512,
#             kernel_size=2,
#             stride=2
#         )
#
#         self.dec4 = DoubleConv(1024, 512)
#
#         self.up3 = nn.ConvTranspose2d(
#             512,
#             256,
#             kernel_size=2,
#             stride=2
#         )
#
#         self.dec3 = DoubleConv(512, 256)
#
#         self.up2 = nn.ConvTranspose2d(
#             256,
#             128,
#             kernel_size=2,
#             stride=2
#         )
#
#         self.dec2 = DoubleConv(256, 128)
#
#         self.up1 = nn.ConvTranspose2d(
#             128,
#             64,
#             kernel_size=2,
#             stride=2
#         )
#
#         self.dec1 = DoubleConv(128, 64)
#
#         # =====================================
#         # Output
#         # =====================================
#
#         self.out_conv = nn.Conv2d(
#             64,
#             num_classes,
#             kernel_size=1
#         )
#
#     def forward(self, x):
#
#         # =====================================
#         # Encoder
#         # =====================================
#
#         x1 = self.input_layer(x)          # [B,64,H/2,W/2]
#
#         x2 = self.maxpool(x1)             # [B,64,H/4,W/4]
#
#         e1 = self.encoder1(x2)            # [B,64,H/4,W/4]
#
#         e2 = self.encoder2(e1)            # [B,128,H/8,W/8]
#
#         e3 = self.encoder3(e2)            # [B,256,H/16,W/16]
#
#         e4 = self.encoder4(e3)            # [B,512,H/32,W/32]
#
#         # =====================================
#         # Bottleneck
#         # =====================================
#
#         b = self.bottleneck(e4)
#
#         # =====================================
#         # Decoder
#         # =====================================
#
#         d4 = self.up4(b)
#         d4 = torch.cat([d4, e4], dim=1)
#         d4 = self.dec4(d4)
#
#         d3 = self.up3(d4)
#         d3 = torch.cat([d3, e3], dim=1)
#         d3 = self.dec3(d3)
#
#         d2 = self.up2(d3)
#         d2 = torch.cat([d2, e2], dim=1)
#         d2 = self.dec2(d2)
#
#         d1 = self.up1(d2)
#         d1 = torch.cat([d1, e1], dim=1)
#         d1 = self.dec1(d1)
#
#         logits = self.out_conv(d1)
#
#         return logits
