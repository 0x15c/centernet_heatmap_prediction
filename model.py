# model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Design considerations:

We use ResNet-9 as backbone of CenterNet. 
ResNet-9 is a lightweight feature extraction network, making our weights very small (~11.5MB).
Furthermore, the inference latency is optimized using lightweight model.
'''

class ResNet9Backbone(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)  # /2

        # Res block @128
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)

        # to 256
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)

        # Res block @256
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # /2

        identity = x
        out = F.relu(self.bn3(self.conv3(x)))
        out = self.bn4(self.conv4(out))
        x = F.relu(out + identity)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)  # /4 total

        identity = x
        out = F.relu(self.bn6(self.conv6(x)))
        out = self.bn7(self.conv7(out))
        x = F.relu(out + identity)

        x = F.relu(self.bn8(self.conv8(x)))
        return x  # (N, 256, 128, 128) for 512 input


class CenterNetModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = ResNet9Backbone(in_channels=3)
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1, 1, 0),
        )

        # init: encourage low initial probs
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # CenterNet-style prior for heatmap: sigmoid(bias) ~ 0.1
        nn.init.constant_(self.head[-1].bias, -2.19)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat) # logits
