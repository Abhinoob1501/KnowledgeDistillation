import torch
import torch.nn as nn
from transformers import ViTModel

class BackboneModel(nn.Module):
    def __init__(self, num_classes: int = 2, backbone: str = 'vit_base_patch16_224'):
        super().__init__()
        
        self.backbone_type = backbone

        if backbone == 'vit_base_patch16_224':
            self.backbone = ViTModel.from_pretrained(
                'google/vit-base-patch16-224', 
                add_pooling_layer=False
            )
            self.feature_dim = self.backbone.config.hidden_size
        else:
            raise ValueError(f'Only vit_base_patch16_224 is supported, got: {backbone}')

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.get_features(x)
        return self.classifier(features)

    def get_features(self, x):
        # Transformer-based models (ViT)
        outputs = self.backbone(pixel_values=x)
        
        # Check if pooler_output exists and is not None
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        else:
            # Use CLS token (first token) from last_hidden_state
            return outputs.last_hidden_state[:, 0]

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
