import torch.nn as nn
from torchvision import models

class ResNet50Model:
    def __init__(self, num_classes=1000):
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def get_model(self):
        return self.model