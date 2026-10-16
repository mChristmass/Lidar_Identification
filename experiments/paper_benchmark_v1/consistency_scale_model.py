"""Loss-scale control: class-balanced gradient, pixel-mean scalar magnitude."""
import torch
from .paired_model import PairedCMX, perturb_view, safe_ce, CONS_WEIGHT, RAMP_EPOCHS
from .consistency_refine_model import backward_objective as pixel_backward

EPS = 1e-8


class ScaleCMX(PairedCMX):
    def __init__(self, method, *args, **kwargs):
        if method not in ('PIXEL', 'SCALE'):
            raise ValueError(method)
        super().__init__('CONS', *args, **kwargs)


def matched_consistency(student, reference, labels):
    teacher = reference.detach().float().softmax(1)
    confidence, prediction = teacher.max(1)
    eligible = (labels != 255) & (prediction == labels) & (confidence >= .5)
    kl = (teacher * (teacher.clamp_min(1e-7).log()
                    - student.float().log_softmax(1))).sum(1)
    if eligible.any():
        terms = [kl[eligible & (labels == c)].mean() for c in labels[eligible].unique()]
        balanced = torch.stack(terms).mean()
        pixel = kl[eligible].mean()
    else:
        balanced = student.float().sum()*0
        pixel = balanced
    # Stop BOTH scalar statistics: allowing ratio gradients would collapse to pixel KL.
    # At numerical-zero/negative KL, suppress the auxiliary term instead of dividing.
    ratio = torch.where(balanced.detach() > EPS,
                        pixel.detach().clamp_min(0)/balanced.detach().clamp_min(EPS),
                        balanced.detach()*0)
    scaled = balanced*ratio
    return scaled, balanced, pixel, ratio, eligible


def backward_objective(model, x, y, dataset, method, epoch, view_seed, scaler, weight):
    if method == 'PIXEL':
        # Exact existing PIXEL objective, not a reimplementation.
        return pixel_backward(model,x,y,dataset,method,epoch,view_seed,scaler,weight)
    if method != 'SCALE':
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
    del reference,ce_reference
    secondary,hidden = perturb_view(x,dataset,view_seed)
    with torch.autocast(device_type=x.device.type, enabled=amp):
        prediction = model(secondary)
        ce_secondary = safe_ce(prediction['logits'],y)
        cons,balanced,pixel,ratio,eligible = matched_consistency(prediction['logits'],target,y)
        objective = .5*ce_secondary+min(1.,epoch/RAMP_EPOCHS)*CONS_WEIGHT*cons
    if not torch.isfinite(objective):
        raise FloatingPointError('Nonfinite scaled objective')
    scaler.scale(objective*weight).backward()
    foreground = (y==1) if dataset=='private' else torch.zeros_like(eligible)
    return dict(loss=.5*ce_ref_value+float(objective.detach()),ce_reference=ce_ref_value,
                ce_secondary=float(ce_secondary.detach()),consistency=float(cons.detach()),
                balanced_kl=float(balanced.detach()),pixel_kl=float(pixel.detach()),
                scale_ratio=float(ratio),scale_match_abs_error=float((cons.detach()-pixel.detach()).abs()),
                effective_cons_weight=CONS_WEIGHT*min(1.,epoch/RAMP_EPOCHS)*float(ratio),
                confidence_coverage=float(eligible.float().mean()),
                selected_foreground_pixels_per_image=float((eligible & foreground).sum())/len(x),
                foreground_pixels_per_image=float(foreground.sum())/len(x),
                hidden_fraction=float(hidden.float().mean()),completion=0.)
