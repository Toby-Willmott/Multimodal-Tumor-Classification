"""Register and export the paired classifier for timm model creation."""

from .baseline import BaselineClassifier, baseline_classifier
from .wsi_only import WSIOnlyClassifier, wsi_only_classifier
from .rna_only import RNAOnlyClassifier, rna_only_classifier

__all__ = ["BaselineClassifier", "baseline_classifier", 
           "WSIOnlyClassifier", "wsi_only_classifier", 
           "RNAOnlyClassifier", "rna_only_classifier"]
