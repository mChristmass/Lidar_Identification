import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.30, beta=0.70, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
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

        tp = (probs * labels).sum(dim=(1, 2))
        fp = (probs * (1.0 - labels)).sum(dim=(1, 2))
        fn = ((1.0 - probs) * labels).sum(dim=(1, 2))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return 1 - tversky.mean()


class Stage2Loss(nn.Module):
    """
    Recommended Stage2 loss:
    Weighted CE + Tversky

    Stage2 target:
    - keep recall during refinement
    - suppress false positives without over-shrinking the target
    - improve region overlap
    """

    def __init__(
        self,
        target_weight=3.0,
        ce_weight=1.0,
        tversky_weight=1.0,
        tversky_alpha=0.30,
        tversky_beta=0.70
    ):
        super().__init__()

        weights = torch.tensor([1.0, target_weight], dtype=torch.float32)
        self.register_buffer("weights", weights)

        self.ce = nn.CrossEntropyLoss(
            weight=self.weights,
            reduction="none"
        )

        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight

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

        tversky_loss = self.tversky(logits, labels, valid_mask)

        total_loss = self.ce_weight * ce_loss + self.tversky_weight * tversky_loss

        return total_loss
