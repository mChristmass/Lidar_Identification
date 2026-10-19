"""Fast CPU checks for pseudo targets and matched-model initialization."""
import torch
from torch.nn import functional as F

from .pseudo_intensity_model import PseudoIntensityCMX, pseudo_target, objective
from .task_control_model import TaskControlCMX


def main():
    x=torch.rand(1,3,16,16)
    target=pseudo_target(x[:,:1])
    assert target.shape==x[:,:1].shape
    assert torch.allclose(pseudo_target(torch.ones_like(x[:,:1])),torch.ones_like(x[:,:1]))
    assert not target.requires_grad
    assert not torch.equal(target,x[:,:1])
    assert target.min()>=0 and target.max()<=1
    classes=2
    y=torch.randint(classes,(1,16,16))
    for method,control in (('FIXED','B0'),('PSEUDO_SUP','R_TASK')):
        torch.manual_seed(42)
        old=TaskControlCMX(control,'CMX_B1',classes,128,256)
        torch.manual_seed(42)
        new=PseudoIntensityCMX(method,'CMX_B1',classes,128,256)
        assert old.state_dict().keys()==new.state_dict().keys()
        assert all(torch.equal(old.state_dict()[k],new.state_dict()[k]) for k in old.state_dict())
        # Do not run a large CMX forward during this fast control check.
        logits=torch.randn(1,classes,16,16,requires_grad=True)
        prediction={'logits':logits}
        if method=='PSEUDO_SUP':
            corrected=(x[:,:1]+new.adapters(x)).clamp(0,1)
            prediction.update(correction_penalty=(corrected-x[:,:1]).square().mean(),
                              pseudo_loss=F.mse_loss(corrected,target))
        loss,_=objective(prediction,y,method)
        loss.backward()
        assert torch.isfinite(loss)
        del old,new
    print('PSEUDO_TEST_OK',flush=True)


if __name__=='__main__':
    main()
