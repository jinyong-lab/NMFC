# Last updated: 2026-04-15 18:30
import torch
import torch.nn as nn
from torchvision import models


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Remove the final classification layer → output is 512D
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)  # (batch_size, 512)
        return x


def get_backbone(device, pretrained=True, freeze=True):
    model = ResNet18Backbone(pretrained=pretrained, freeze=freeze)
    model = model.to(device)
    model.eval()
    return model
