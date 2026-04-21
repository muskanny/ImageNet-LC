"""
ImageNet-LC: Localized Corruption benchmark pipeline.

Four stages, each independently runnable via ``main.py``:

    Stage 1: Bounding-box generation    (imagenet_lc.stages.stage1_bboxes)
    Stage 2: Corruption application     (imagenet_lc.stages.stage2_corrupt)
    Stage 3: Inference                  (imagenet_lc.stages.stage3_inference)
    Stage 4: Evaluation                 (imagenet_lc.stages.stage4_evaluate)

Public API:
    apply_corruption, SUPPORTED_CORRUPTIONS    -- corruption dispatcher
    YOLOLocalizer                              -- Phase 1 detector wrapper
    load_model, list_models                    -- Stage 3 model registry
"""

from .pipeline import apply_corruption, SUPPORTED_CORRUPTIONS
from .localize import YOLOLocalizer
from .models import load_model, list_models

__all__ = [
    "apply_corruption",
    "SUPPORTED_CORRUPTIONS",
    "YOLOLocalizer",
    "load_model",
    "list_models",
]
__version__ = "1.0.0"
