"""Paper backbones and their classification head."""

import torch
from enccluster.architectures import ResNet20, ConvMixer256_8, ConvNeXT


class MobileNetV2(torch.nn.Module):
    """Official torchvision MobileNetV2, width multiplier 1.0, from scratch."""

    embeddings_dim = 1280

    def __init__(self):
        super().__init__()
        from torchvision.models import mobilenet_v2

        self.features = mobilenet_v2(weights=None).features
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        return self.dropout(self.features(x).mean(dim=(2, 3)))


class Model(torch.nn.Module):
    def __init__(self, model, num_classes=10):
        super().__init__()
        backbones = {
            "resnet20": ResNet20,
            "convmixer256_8": ConvMixer256_8,
            "mobilenet_v2": MobileNetV2,
            "convnext": ConvNeXT,
        }
        if model not in backbones:
            raise ValueError(f"Unknown architecture: {model}")
        self.backbone = backbones[model]()
        self.classifier = torch.nn.Linear(self.backbone.embeddings_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x))
