import torch
import torch.nn as nn
from torchvision import models

class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        # Carichiamo una ResNet pre-addestrata (apprende già a distinguere forme e texture)
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modifichiamo l'ultimo strato (FC) per adattarlo alle nostre 2 classi (Real vs Fake)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def get_model(device):
    model = DeepfakeDetector()
    model = model.to(device)
    return model