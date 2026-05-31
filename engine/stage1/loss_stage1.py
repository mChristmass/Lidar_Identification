# import torch
# import torch.nn as nn
#
#
# # class SegmentationLoss(nn.Module):
# #
# #     def __init__(self, target_weight=10):
# #
# #         super().__init__()
# #
# #         weights = torch.tensor([1.0, target_weight])
# #
# #         # 注册buffer（自动跟随device）
# #         self.register_buffer("weights", weights)
# #
# #         self.ce = nn.CrossEntropyLoss(
# #             weight=self.weights,
# #             reduction="none"
# #         )
# #
# #     def forward(self, logits, labels, valid_mask=None):
# #
# #         loss = self.ce(logits, labels)
# #
# #         if valid_mask is not None:
# #
# #             loss = loss * valid_mask
# #
# #             return loss.sum() / (valid_mask.sum() + 1e-6)
# #
# #         return loss.mean()
#
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super().__init__()
#         self.smooth = smooth
#
#     def forward(self, logits, labels, valid_mask=None):
#         # logits: [B, C, H, W]
#         # labels: [B, H, W]
#         probs = torch.softmax(logits, dim=1)[:, 1, :, :]   # 前景概率
#         labels = labels.float()
#
#         if valid_mask is not None:
#             probs = probs * valid_mask
#             labels = labels * valid_mask
#
#         intersection = (probs * labels).sum(dim=(1, 2))
#         union = probs.sum(dim=(1, 2)) + labels.sum(dim=(1, 2))
#
#         dice = (2 * intersection + self.smooth) / (union + self.smooth)
#         return 1 - dice.mean()
#
#
# class SegmentationLoss(nn.Module):
#     def __init__(self, target_weight=10, dice_weight=1.0, ce_weight=1.0):
#         super().__init__()
#
#         weights = torch.tensor([1.0, target_weight])
#         self.register_buffer("weights", weights)
#
#         self.ce = nn.CrossEntropyLoss(
#             weight=self.weights,
#             reduction="none"
#         )
#
#         self.dice = DiceLoss()
#         self.dice_weight = dice_weight
#         self.ce_weight = ce_weight
#
#     def forward(self, logits, labels, valid_mask=None):
#         ce_loss = self.ce(logits, labels)   # [B, H, W]
#
#         if valid_mask is not None:
#             ce_loss = ce_loss * valid_mask
#             ce_loss = ce_loss.sum() / (valid_mask.sum() + 1e-6)
#         else:
#             ce_loss = ce_loss.mean()
#
#         dice_loss = self.dice(logits, labels, valid_mask)
#
#         total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
#         return total_loss




import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, labels, valid_mask=None):
        """
        logits: [B, 2, H, W]
        labels: [B, H, W], value: 0/1
        valid_mask: [B, H, W], value: 0/1, optional
        """

        probs = torch.softmax(logits, dim=1)[:, 1, :, :]  # foreground probability
        labels = labels.float()

        if valid_mask is not None:
            valid_mask = valid_mask.float()
            probs = probs * valid_mask
            labels = labels * valid_mask

        intersection = (probs * labels).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + labels.sum(dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """
    Recommended Stage1 loss:
    Weighted CE + 0.5 * Dice

    Stage1 target:
    - keep high recall
    - avoid excessive false positives
    """

    def __init__(
        self,
        target_weight=5.0,
        ce_weight=1.0,
        dice_weight=0.5
    ):
        super().__init__()

        weights = torch.tensor([1.0, target_weight], dtype=torch.float32)
        self.register_buffer("weights", weights)

        self.ce = nn.CrossEntropyLoss(
            weight=self.weights,
            reduction="none"
        )

        self.dice = SoftDiceLoss()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, labels, valid_mask=None):
        """
        logits: [B, 2, H, W]
        labels: [B, H, W]
        valid_mask: [B, H, W], optional
        """

        ce_loss = self.ce(logits, labels)

        if valid_mask is not None:
            valid_mask = valid_mask.float()
            ce_loss = ce_loss * valid_mask
            ce_loss = ce_loss.sum() / (valid_mask.sum() + 1e-6)
        else:
            ce_loss = ce_loss.mean()

        dice_loss = self.dice(logits, labels, valid_mask)

        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return loss