import torch
import torch.nn as nn


class EcoSortCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 64 x 64 -> 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 32 x 32 -> 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 16 x 16 -> 8 x 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # 128 x 8 x 8 = 8192
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),

            # 4 EcoSort classes
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x