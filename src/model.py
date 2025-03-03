import torch
import torch.nn as nn
import torchvision.models as models

class CataractPhaseModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=True)
        # Replace final layer for our 4 classes
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, num_classes)

    def forward(self, x):
        return self.backbone(x)

def build_model(num_classes=4):
    return CataractPhaseModel(num_classes=num_classes)
