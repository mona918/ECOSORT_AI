import torch
import torch.nn as nn
from torchvision import models


class EcoSortResNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        # Fine-tune layer3
        for param in self.model.layer3.parameters():
            param.requires_grad = True

        # Fine-tune layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # New classifier
        num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            num_features,
            4
        )

    def forward(self, x):

        return self.model(x)
    