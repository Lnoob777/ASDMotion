import torch
<<<<<<< HEAD

class MicroKinetics:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def compute(self, s: torch.Tensor, mask: torch.Tensor):
        v = torch.zeros_like(s)
        a = torch.zeros_like(s)

        # Velocity
        v[1:] = s[1:] - s[:-1]
        valid_v = mask & torch.roll(mask, 1)
        valid_v[0] = False
        v *= valid_v[:, None]

        # Acceleration
        a[2:] = s[2:] - 2 * s[1:-1] + s[:-2]
        valid_a = mask & torch.roll(mask, 1) & torch.roll(mask, 2)
        valid_a[:2] = False
        a *= valid_a[:, None]

        # Energy
        e = torch.norm(v, dim=1) + self.eps
        e *= valid_v

        return v, a, e
=======
class MicroKinetics:
    def __init__(self,eps:float=1e-6):
        self.eps=eps
    def compute(self,s:torch.Tensor,mask:torch.Tensor):
        T,D=s.shape
        v = torch.zeros_like(s)
        a =torch.zeros_like(s)
        v[1:]=s[1:]-s[0:-1]
        a[2:]=v[2:]-v[1:-1]
        e = torch.norm(v,dim=1)
        v =v*mask[:,None]
        a=a*mask[:,None]
        e=e*mask
        return{
            "velocity":v,
            "acceleration":a,
            "energy":e
        }
    
>>>>>>> 536151f8805e54555e46a5c1f93506346c3db8fd
