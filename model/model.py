import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class EffientNetB0(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch", "efficientnet_b0", pretrained=True
        )
        for param in self.model.conv_stem.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, 512), nn.Dropout(0.2), nn.ReLU(), nn.Linear(512, 39)
        )

    def forward(self, x):
        return self.model(x)


class MixNetM(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch", "mixnet_m", pretrained=True
        )
        for param in self.model.conv_stem.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Linear(1536, 512), nn.Dropout(0.2), nn.ReLU(), nn.Linear(512, 39)
        )

    def forward(self, x):
        return self.model(x)
