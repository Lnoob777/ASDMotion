import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

class PoseN(nn.Module):
    def __init__(self,pretrained : bool=True):
        super().__init__()

        backbone = resnet18(pretrained=pretrained)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.proj = nn.Sequential(
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

    def forward(self,x):
        x=self.backbone(x)
        x=x.proj(x)
        return x
