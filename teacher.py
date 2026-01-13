import torch
import torch.nn as nn
from backbone import BackboneModel
from config import Config

class TeacherModel(nn.Module):
    def __init__(self, group_id, backbone, config=False):
        super().__init__()
        self.config = Config.teacher_config if not config else config
        self.backbone = backbone
        
        # BackboneModel always has feature_dim and num_classes
        feature_dim = backbone.feature_dim
        num_classes = backbone.num_classes
            
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone.get_features(x)
        features = features.to(self.classifier.weight.device)
        output = self.classifier(features)
        return output