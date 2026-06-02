import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, labels, valid_mask=None):
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]
        labels = labels.float()

        if valid_mask is not None:
            valid_mask = valid_mask.float()
            probs = probs * valid_mask
            labels = labels * valid_mask

        intersection = (probs * labels).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + labels.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.35, beta=0.65, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, labels, valid_mask=None):
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]
        labels = labels.float()

        if valid_mask is not None:
            valid_mask = valid_mask.float()
            probs = probs * valid_mask
            labels = labels * valid_mask

        tp = (probs * labels).sum(dim=(1, 2))
        fp = (probs * (1.0 - labels)).sum(dim=(1, 2))
        fn = ((1.0 - probs) * labels).sum(dim=(1, 2))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class SegmentationLoss(nn.Module):
    """
    Selectable Stage1 loss.

    loss_type:
    - "ce_dice": Weighted CE + Dice, the original Stage1 baseline.
    - "ce_tversky": Weighted CE + Tversky, recall-biased variant.
    """

    def __init__(
        self,
        loss_type="ce_dice",
        target_weight=5.0,
        ce_weight=1.0,
        dice_weight=0.5,
        tversky_weight=0.5,
        tversky_alpha=0.35,
        tversky_beta=0.65,
    ):
        super().__init__()

        self.loss_type = loss_type
        weights = torch.tensor([1.0, target_weight], dtype=torch.float32)
        self.register_buffer("weights", weights)

        self.ce = nn.CrossEntropyLoss(weight=self.weights, reduction="none")
        self.dice = SoftDiceLoss()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight

        if loss_type not in {"ce_dice", "ce_tversky"}:
            raise ValueError(f"Unsupported Stage1 loss_type: {loss_type}")

    def forward(self, logits, labels, valid_mask=None):
        ce_loss = self.ce(logits, labels)

        if valid_mask is not None:
            valid_mask = valid_mask.float()
            ce_loss = ce_loss * valid_mask
            ce_loss = ce_loss.sum() / (valid_mask.sum() + 1e-6)
        else:
            ce_loss = ce_loss.mean()

        if self.loss_type == "ce_dice":
            region_loss = self.dice(logits, labels, valid_mask)
            region_weight = self.dice_weight
        else:
            region_loss = self.tversky(logits, labels, valid_mask)
            region_weight = self.tversky_weight

        return self.ce_weight * ce_loss + region_weight * region_loss
