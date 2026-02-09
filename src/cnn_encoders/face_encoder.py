import torch 
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

class FaceN(nn.Module):
    def __init__(self,pretrained : bool=True):
        super().__init__()

        backbone = resnet18(pretrained=pretrained)
        backbone.fc = nn.Identity()
