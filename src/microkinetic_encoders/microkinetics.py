import torch
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
    