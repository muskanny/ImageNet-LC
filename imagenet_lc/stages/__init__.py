"""Pipeline stages for ImageNet-LC."""

from . import stage1_bboxes
from . import stage2_corrupt
from . import stage3_inference
from . import stage4_evaluate

__all__ = [
    "stage1_bboxes",
    "stage2_corrupt",
    "stage3_inference",
    "stage4_evaluate",
]
