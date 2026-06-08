import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.stage1.loss_stage1 import SoftDiceLoss


class ErrorFocusedRefineLoss(nn.Module):
    def __init__(
        self,
        target_weight=3.0,
        ce_weight=1.0,
        dice_weight=1.0,
        error_pixel_weight=5.0,
        correct_pixel_weight=1.0,
        correct_delta_reg_weight=0.05,
    ):
        super().__init__()
        class_weights = torch.tensor([1.0, target_weight], dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)
        self.dice = SoftDiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.error_pixel_weight = error_pixel_weight
        self.correct_pixel_weight = correct_pixel_weight
        self.correct_delta_reg_weight = correct_delta_reg_weight

    def forward(self, final_logits, delta_logits, base_logits, labels, roi_mask, base_threshold):
        roi_mask = roi_mask.float()
        with torch.no_grad():
            base_prob = torch.softmax(base_logits, dim=1)[:, 1]
            base_pred = base_prob > base_threshold
            error_mask = base_pred != labels.bool()
            focus_weights = torch.where(
                error_mask,
                torch.full_like(roi_mask, self.error_pixel_weight),
                torch.full_like(roi_mask, self.correct_pixel_weight),
            )
            focus_weights = focus_weights * roi_mask

        ce_map = F.cross_entropy(
            final_logits,
            labels,
            weight=self.class_weights,
            reduction="none",
        )
        ce_loss = (ce_map * focus_weights).sum() / (focus_weights.sum() + 1e-6)
        dice_loss = self.dice(final_logits, labels, valid_mask=roi_mask)

        correct_roi = (~error_mask).float() * roi_mask
        delta_magnitude = delta_logits.abs().mean(dim=1)
        preserve_loss = (delta_magnitude * correct_roi).sum() / (correct_roi.sum() + 1e-6)

        return (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
            + self.correct_delta_reg_weight * preserve_loss
        )
