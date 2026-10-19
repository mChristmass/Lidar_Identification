"""Pseudo-clean intensity controls; the target is only a smoothed observation."""
import torch
from torch import nn
from torch.nn import functional as F

from .model import CMXPaper
from .task_control_model import TaskControlCMX, CORRECTION_WEIGHT

PSEUDO_WEIGHT = 1.0


def pseudo_target(intensity):
    one = intensity.new_tensor([1., 2., 1.])
    kernel = (one[:, None] * one[None, :]).reshape(1, 1, 3, 3) / 16
    blurred = F.conv2d(F.pad(intensity.float(), (1, 1, 1, 1), mode='replicate'), kernel.float())
    return .5 * intensity.float() + .5 * blurred


class PseudoIntensityCMX(TaskControlCMX):
    def __init__(self, method, variant='CMX_B2', num_classes=40, image_size=640, decoder_dim=256):
        if method not in ('FIXED', 'PSEUDO_SUP'):
            raise ValueError(method)
        super().__init__('B0' if method == 'FIXED' else 'R_TASK',
                         variant, num_classes, image_size, decoder_dim)
        self.pseudo_method = method

    def forward(self, inputs):
        if self.pseudo_method == 'FIXED':
            smoothed = pseudo_target(inputs[:, :1]).to(inputs.dtype)
            return CMXPaper.forward(self, torch.cat((smoothed, inputs[:, 1:]), 1))
        delta = self.adapters(inputs)
        intensity = inputs[:, :1]
        corrected = (intensity + delta).clamp(0, 1)
        prediction = CMXPaper.forward(self, torch.cat((corrected, inputs[:, 1:]), 1))
        prediction['correction_penalty'] = delta.float().square().mean()
        if self.training:
            target = pseudo_target(intensity.detach())
            prediction['pseudo_loss'] = (corrected.float() - target).square().mean()
        return prediction


def objective(prediction, labels, method):
    ce = F.cross_entropy(prediction['logits'].float(), labels, ignore_index=255)
    if method == 'FIXED':
        return ce, dict(ce=float(ce.detach()), correction_l2=0., pseudo_mse=0.)
    penalty = prediction['correction_penalty']
    pseudo = prediction['pseudo_loss']
    total = ce + CORRECTION_WEIGHT * penalty + PSEUDO_WEIGHT * pseudo
    return total, dict(ce=float(ce.detach()), correction_l2=float(penalty.detach()),
                       pseudo_mse=float(pseudo.detach()))
