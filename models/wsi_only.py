from torch import nn
from timm.models import register_model

from .baseline import AttentionPool

class WSIOnlyClassifier(nn.Module):
    def __init__(self, wsi_embed_dim=768, embed_dim=768, num_classes=2, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.wsi_encoder = AttentionPool(wsi_embed_dim, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def get_classifier(self):
        return self.head

    def forward(self, x):
        embedding = self.wsi_encoder(x)
        return self.head(embedding)

@register_model
def wsi_only_classifier(pretrained=False, **kwargs):

    if pretrained:
        raise ValueError("No pretrained weights are supplied for this model")
    for name in ("pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"):
        kwargs.pop(name, None)
    return WSIOnlyClassifier(**kwargs)