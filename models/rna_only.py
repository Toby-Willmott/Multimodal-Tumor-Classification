from torch import nn
from timm.models import register_model

class RNAOnlyClassifier(nn.Module):
    def __init__(self, rna_embed_dim, embed_dim=768, num_classes=2, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        rna_layers = []
        for in_features in (rna_embed_dim, embed_dim):
            rna_layers.extend([
                nn.Linear(in_features, embed_dim), nn.LayerNorm(embed_dim),
                nn.ReLU(), nn.Dropout(0.2),
            ])
        rna_layers.append(nn.Linear(embed_dim, embed_dim))
        self.rna_encoder = nn.Sequential(*rna_layers)
        self.head = nn.Linear(embed_dim, num_classes)

    def get_classifier(self):
        return self.head

    def forward(self, x):
        rna, _ = x
        embedding = self.rna_encoder(rna)
        return self.head(embedding)


@register_model
def rna_only_classifier(pretrained=False, **kwargs):
    if pretrained:
        raise ValueError("The model has no pretrained weights")
    # timm passes these metadata keys even though this model does not use them.
    for name in ("pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"):
        kwargs.pop(name, None)
    return RNAOnlyClassifier(**kwargs)
