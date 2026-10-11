"""Observed-view supervision: no clean image, physical posterior or final-test data."""
import torch
from torch import nn
from torch.nn import functional as F

from .model import BACKBONES
from .task_control_model import TaskControlCMX

METHODS = ('B0', 'S_RAW', 'PAIR', 'CONS', 'MASK', 'BOTH')
CONS_WEIGHT = .2
MASK_WEIGHT = .1
RAMP_EPOCHS = 5
VIEW_CONFIG = {'private': {'hide_probability': .10, 'noise_std': .01, 'block': 8},
               'nyuv2': {'hide_probability': .25, 'noise_std': .03, 'block': 16}}


class PairedCMX(TaskControlCMX):
    def __init__(self, method, variant='CMX_B2', num_classes=40, image_size=640, decoder_dim=256):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__('B0' if method == 'B0' else 'S_RAW',
                         variant, num_classes, image_size, decoder_dim)
        self.paired_method = method
        self.capture_features = False
        self._captured = None
        if method in ('MASK', 'BOTH'):
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed((torch.initial_seed()+2903) % (2**63))
                self.adapters['completion'] = nn.ModuleList([
                    nn.Sequential(nn.Conv2d(c, 64, 3, padding=1), nn.GroupNorm(4, 64),
                                  nn.GELU(), nn.Conv2d(64, c, 1))
                    for c in BACKBONES[variant][1][:2]])
        self.decode_head.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        if self.capture_features:
            self._captured = tuple(args[0][:2])

    def forward(self, inputs):
        self._captured = None
        try:
            result = super().forward(inputs)
            if self.capture_features:
                result['features'] = self._captured
            return result
        finally:
            self._captured = None


def perturb_view(x, dataset, seed):
    """Same deterministic secondary view for PAIR/CONS/MASK/BOTH, no label access."""
    config = VIEW_CONFIG[dataset]
    rng = torch.Generator(device=x.device).manual_seed(seed)
    b, _, h, w = x.shape
    block = config['block']
    coarse = torch.rand((b, 1, (h+block-1)//block, (w+block-1)//block),
                        device=x.device, generator=rng) < config['hide_probability']
    hidden = coarse.repeat_interleave(block, -2).repeat_interleave(block, -1)[..., :h, :w]
    hidden = hidden & (x[:, 2:3] > .5)
    out = x.clone()
    noise = torch.randn((b, 1, h, w), device=x.device, generator=rng)
    out[:, :1] = (x[:, :1] + config['noise_std']*noise).clamp(0, 1)
    out[:, 1:2] = x[:, 1:2].masked_fill(hidden, 0)
    out[:, 2:3] = x[:, 2:3].masked_fill(hidden, 0)
    return out, hidden


def safe_ce(logits, labels):
    valid = labels != 255
    if not valid.any():
        return logits.float().sum()*0
    return F.cross_entropy(logits.float(), labels, ignore_index=255)


def consistency_loss(student, reference, labels, force_support=False):
    """Class-balanced KL over correct, confident reference pixels; reference detached."""
    teacher = reference.detach().float().softmax(1)
    confidence, prediction = teacher.max(1)
    valid = (labels != 255) & (prediction == labels) & (confidence >= .5)
    if force_support:
        # Synthetic timing/peak-memory check only; NEVER enabled by real training.
        valid = labels != 255
    kl = (teacher * (teacher.clamp_min(1e-7).log()
                    - student.float().log_softmax(1))).sum(1)
    terms = [kl[valid & (labels == category)].mean()
             for category in labels[valid].unique()]
    loss = torch.stack(terms).mean() if terms else student.float().sum()*0
    return loss, valid.float().mean()


def completion_loss(model, student_features, reference_features, hidden, labels):
    """Predict stopped observed fused features only where valid depth was hidden.

    Fractional area weights at two feature scales exclude original holes/ignored labels.
    This is NOT supervision of actual missing-depth values.
    """
    supported = hidden.float() * (labels[:, None] != 255)
    terms = []
    for head, feature, target in zip(model.adapters['completion'],
                                     student_features, reference_features):
        pred = head(feature).float()
        weights = F.adaptive_avg_pool2d(supported, pred.shape[-2:])[:, 0]
        target = target.detach().float()
        distance = 1-F.cosine_similarity(pred, target, dim=1, eps=1e-6)
        terms.append((distance*weights).sum()/weights.sum().clamp_min(1e-6))
    return torch.stack(terms).mean()


def backward_objective(model, x, y, dataset, method, epoch, view_seed, scaler, weight,
                       calibrate=False):
    """Sequential reference/student backward limits peak memory; no optimizer step here."""
    paired = method not in ('B0', 'S_RAW')
    use_mask = method in ('MASK', 'BOTH')
    use_cons = method in ('CONS', 'BOTH')
    amp = x.device.type == 'cuda'
    ramp = min(1., epoch/RAMP_EPOCHS)
    model.capture_features = use_mask
    try:
        with torch.autocast(device_type=x.device.type, enabled=amp):
            reference = model(x)
            ce_reference = safe_ce(reference['logits'], y)
        if not torch.isfinite(ce_reference):
            raise FloatingPointError('Nonfinite reference CE')
        target_logits = reference['logits'].detach() if use_cons else None
        targets = tuple(f.detach() for f in reference['features']) if use_mask else None
        scaler.scale(ce_reference*weight*(.5 if paired else 1.)).backward()
        ce_ref_value = float(ce_reference.detach())
        del reference, ce_reference
        if not paired:
            return dict(loss=ce_ref_value, ce_reference=ce_ref_value, ce_secondary=0.,
                        consistency=0., completion=0., confidence_coverage=0., hidden_fraction=0.)
        secondary, hidden = perturb_view(x, dataset, view_seed)
        with torch.autocast(device_type=x.device.type, enabled=amp):
            prediction = model(secondary)
            ce_secondary = safe_ce(prediction['logits'], y)
            cons, coverage = (consistency_loss(prediction['logits'], target_logits, y,
                                               force_support=calibrate)
                              if use_cons else (ce_secondary*0, ce_secondary.detach()*0))
            recover = (completion_loss(model, prediction['features'], targets, hidden, y)
                       if use_mask else ce_secondary*0)
            objective = .5*ce_secondary + ramp*(CONS_WEIGHT*cons+MASK_WEIGHT*recover)
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite secondary objective')
        scaler.scale(objective*weight).backward()
        return dict(loss=.5*ce_ref_value+float(objective.detach()), ce_reference=ce_ref_value,
                    ce_secondary=float(ce_secondary.detach()), consistency=float(cons.detach()),
                    completion=float(recover.detach()), confidence_coverage=float(coverage.detach()),
                    hidden_fraction=float(hidden.float().mean()))
    finally:
        model.capture_features = False
        model._captured = None
