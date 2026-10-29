"""The baseline multimodal classifier that takes in both WSI and RNA features but relies almost entirely on RNA: the 7 validations errors
included all 6 RNA-only erors while the WSIonly model correctly classifies all ofthem. Concat also seems somewhat of a bad way to combine 
the features. This model randomly hides one modality per patient during trainign to force the model to use the slide information more."""




import torch 
from timm.models import register_model

from .baseline import BaselineClassifier

class ModalityDropoutClassifier(BaselineClassifier):
    def __init__(self, modality_dropout_prob=0.15, **kwargs):
        super().__init__(**kwargs)
        self.modality_dropout_prob = modality_dropout_prob

    def forward(self, x):
        rna, wsi = x
        wsi_embedding = self.wsi_encoder(wsi)
        rna_embedding = self.rna_encoder(rna)

        
        if self.training and self.modality_dropout_prob > 0.0:
            r = torch.rand(rna_embedding.shape[0], device = rna_embedding.device)

            drop_rna = r < self.modality_dropout_prob
            drop_wsi =  (self.modality_dropout_prob <= r)  & ( r < 2 * self.modality_dropout_prob)

            keep_rna = (~drop_rna).to(rna_embedding.dtype).unsqueeze(1)
            keep_wsi = (~drop_wsi).to(wsi_embedding.dtype).unsqueeze(1)

            rna_embedding = rna_embedding * keep_rna
            wsi_embedding = wsi_embedding * keep_wsi

        embedding = torch.cat([wsi_embedding, rna_embedding], dim=-1)
        return self.head(embedding)

@register_model
def modality_dropout_classifier(pretrained=False, **kwargs):
    if pretrained:
        raise ValueError("The model has no pretrained weights")
    for name in ("pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"):
            kwargs.pop(name, None)
    return ModalityDropoutClassifier(**kwargs)
