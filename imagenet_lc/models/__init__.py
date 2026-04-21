"""
Unified model interface for inference.

Every architecture listed in Table 1 of the paper is wrapped behind a
common ``BaseClassifier`` API so that the inference stage can treat them
interchangeably. Weights are ImageNet-pretrained and come from torchvision
or timm.

Note
----
The paper's numbers were produced using a mix of Keras and PyTorch
checkpoints (see Muskan's ``Inference_Modules``). This repo consolidates
everything onto PyTorch (torchvision + timm) for simpler dependencies.
Absolute accuracy numbers may therefore differ slightly from those
reported in the paper, but qualitative robustness trends (CNN vs ViT,
CE rankings) should be preserved.
"""

from .base import BaseClassifier, MODEL_REGISTRY, list_models, load_model

# Importing these modules triggers @register_model() for every class.
from . import torch_models  # noqa: F401
from . import timm_models   # noqa: F401

__all__ = ["BaseClassifier", "MODEL_REGISTRY", "list_models", "load_model"]
