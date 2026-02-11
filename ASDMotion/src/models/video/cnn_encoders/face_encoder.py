import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class FDataset(Dataset):
    def __init__(self,direc,transform=None):
        self.direc = direc
        self.transform = transform
        self.image_paths = []

        for video in sorted(os.listdir(direc)):
            vdirec = os.path.join(direc,video)
            if not (os.path.isdir(vdirec)):
                continue
            for img in sorted(os.listdir(vdirec)):
                if img.endswith(".png"):
                    self.image_paths.append(
                        os.path.join(vdirec, img)
                    )
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),   # [3, 224, 224]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class FaceN(nn.Module):
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

dataset = FDataset(direc="data/processed/faces",transform=face_transform)
loader = DataLoader(dataset,shuffle=False,batch_size=16,num_workers=4)

model = FaceN(pretrained=True).to(device)
model.eval()

all_features = []

with torch.no_grad():
    for images in loader:
        images = images.to(device)
        features = model(images)   # [B, 256]
        all_features.append(features.cpu())

face_features = torch.cat(all_features, dim=0)
