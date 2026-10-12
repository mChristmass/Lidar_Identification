"""Two isolated consistency hypotheses; unchanged model, views and CE supervision."""
import torch
from .paired_model import PairedCMX, perturb_view, safe_ce, CONS_WEIGHT, RAMP_EPOCHS, VIEW_CONFIG


class RefinedCMX(PairedCMX):
    def __init__(self, method, *args, **kwargs):
        if method not in ('LOCAL', 'PIXEL'):
            raise ValueError(method)
        super().__init__('CONS', *args, **kwargs)


def loss_and_support(student, reference, labels, region, method):
    teacher = reference.detach().float().softmax(1)
    confidence, prediction = teacher.max(1)
    eligible = (labels != 255) & (prediction == labels) & (confidence >= .5)
    selected = eligible & region[:,0] if method == 'LOCAL' else eligible
    kl = (teacher * (teacher.clamp_min(1e-7).log()
                    - student.float().log_softmax(1))).sum(1)
    if method == 'LOCAL':
        # Same class balancing as original CONS, only spatial support changes.
        terms = [kl[selected & (labels == c)].mean() for c in labels[selected].unique()]
        loss = torch.stack(terms).mean() if terms else student.float().sum()*0
    elif method == 'PIXEL':
        # Same support as original CONS, only class-equal -> pixel-equal reduction changes.
        loss = kl[selected].mean() if selected.any() else student.float().sum()*0
    else:
        raise ValueError(method)
    return loss, selected, eligible


def affected_region(reference, secondary, hidden, dataset):
    # Known training-only perturbations, not an inferred physical reliability map.
    changed = (secondary[:,:1]-reference[:,:1]).abs() >= 2*VIEW_CONFIG[dataset]['noise_std']
    return (hidden | changed).detach()


def backward_objective(model, x, y, dataset, method, epoch, view_seed, scaler, weight):
    if method not in ('LOCAL','PIXEL'):
        raise ValueError(method)
    amp = x.device.type == 'cuda'
    with torch.autocast(device_type=x.device.type, enabled=amp):
        reference = model(x)
        ce_reference = safe_ce(reference['logits'], y)
    if not torch.isfinite(ce_reference):
        raise FloatingPointError('Nonfinite reference CE')
    target = reference['logits'].detach()
    scaler.scale(.5*ce_reference*weight).backward()
    ce_ref_value = float(ce_reference.detach())
    del reference, ce_reference
    secondary, hidden = perturb_view(x, dataset, view_seed)
    region = affected_region(x, secondary, hidden, dataset)
    with torch.autocast(device_type=x.device.type, enabled=amp):
        prediction = model(secondary)
        ce_secondary = safe_ce(prediction['logits'], y)
        cons, selected, eligible = loss_and_support(prediction['logits'], target, y, region, method)
        objective = .5*ce_secondary + min(1.,epoch/RAMP_EPOCHS)*CONS_WEIGHT*cons
    if not torch.isfinite(objective):
        raise FloatingPointError('Nonfinite secondary objective')
    scaler.scale(objective*weight).backward()
    foreground = (y == 1) if dataset == 'private' else torch.zeros_like(selected)
    return dict(loss=.5*ce_ref_value+float(objective.detach()), ce_reference=ce_ref_value,
                ce_secondary=float(ce_secondary.detach()), consistency=float(cons.detach()),
                completion=0., confidence_coverage=float(selected.float().mean()),
                eligible_coverage=float(eligible.float().mean()),
                region_fraction=float(region.float().mean()),
                selected_foreground_pixels_per_image=float((selected & foreground).sum())/len(x),
                foreground_pixels_per_image=float(foreground.sum())/len(x),
                hidden_fraction=float(hidden.float().mean()))
