import torch
import torch.nn as nn
import torchvision.models as models

class CataractPhaseModel(nn.Module):
    def __init__(self, num_classes=4):
        """
        Create a MobileNetV2 model with 4 output classes
        (Capsulorhexis, I/A, Phaco, IOL insertion).
        """
        super(CataractPhaseModel, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=True)
        # Replace the final classification layer
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, num_classes)

    def forward(self, x):
        return self.backbone(x)

def build_model(num_classes=4):
    """
    Helper function to build and return the model.
    """
    model = CataractPhaseModel(num_classes=num_classes)
    return model
