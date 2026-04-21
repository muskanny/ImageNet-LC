"""
Corruption functions for ImageNet-LC.

Each function takes an input BGR image (numpy array) and a bounding box
(x_min, y_min, x_max, y_max), along with a severity level (1-5), and
returns the corrupted image.
"""

from .camouflage import apply_camouflaging
from .dust_scratches import apply_dust_scratches
from .fingerprint import apply_fingerprint_noise
from .focus_shift import apply_object_focus_shift
from .illumination import apply_illumination_variation
from .lens_flare import apply_lens_flare
from .occlusion import apply_partial_occlusion

__all__ = [
    "apply_camouflaging",
    "apply_dust_scratches",
    "apply_fingerprint_noise",
    "apply_object_focus_shift",
    "apply_illumination_variation",
    "apply_lens_flare",
    "apply_partial_occlusion",
]
