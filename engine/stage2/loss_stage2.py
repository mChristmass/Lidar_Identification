import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, labels, valid_mask=None):
        """
        logits: [B, 2, H, W]
        labels: [B, H, W], 0/1
        valid_mask: [B, H, W], optional
        """

        probs = torch.softmax(logits, dim=1)[:, 1, :, :]
        labels = labels.float()

        if valid_mask is not None:
            valid_mask = valid_mask.float()
            probs = probs * valid_mask
            labels = labels * valid_mask

        intersection = (probs * labels).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + labels.sum(dim=(1, 2))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class Stage2Loss(nn.Module):
    """
    Recommended Stage2 loss:
    Weighted CE + Dice

    Stage2 target:
    - suppress false positives
    - keep recall
    - improve region overlap
    """

    def __init__(
        self,
        target_weight=2.0,
        ce_weight=1.0,
        dice_weight=1.0
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
        """

        ce_loss = self.ce(logits, labels)

        if valid_mask is not None:
            valid_mask = valid_mask.float()
            ce_loss = ce_loss * valid_mask
            ce_loss = ce_loss.sum() / (valid_mask.sum() + 1e-6)
        else:
            ce_loss = ce_loss.mean()

        dice_loss = self.dice(logits, labels, valid_mask)

        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return total_loss