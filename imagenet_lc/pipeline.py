"""
Pipeline dispatcher for ImageNet-LC corruptions.

This module provides a single entry point ``apply_corruption`` that routes a
corruption name to the appropriate function in ``imagenet_lc.corruptions``.
It also provides ``SUPPORTED_CORRUPTIONS`` for callers (e.g. the CLI) to
validate requested corruption names.

The actual corruption implementations live in ``imagenet_lc.corruptions`` and
are intentionally kept untouched — this module only wires them together.
"""

from .corruptions import (
    apply_camouflaging,
    apply_dust_scratches,
    apply_fingerprint_noise,
    apply_object_focus_shift,
    apply_illumination_variation,
    apply_lens_flare,
    apply_partial_occlusion,
)


SUPPORTED_CORRUPTIONS = [
    "camouflage",
    "dust_scratches",
    "fingerprint",
    "focus_shift",
    "illumination_variation",
    "lens_flare",
    "occlusion",
]


def apply_corruption(
    corruption_name,
    image,
    bbox,
    severity,
    fingerprint_texture=None,
    flare_img=None,
    illumination_mode="shadow",
):
    """
    Apply a single named corruption to an image within a bounding box.

    Parameters
    ----------
    corruption_name : str
        One of ``SUPPORTED_CORRUPTIONS``.
    image : numpy.ndarray
        Input image in BGR format (as read by ``cv2.imread``).
    bbox : tuple of int
        Bounding box ``(x_min, y_min, x_max, y_max)``.
    severity : int
        Severity level in [1, 5].
    fingerprint_texture : numpy.ndarray, optional
        Required when ``corruption_name == 'fingerprint'``. A BGR image loaded
        with ``cv2.imread`` of the fingerprint texture.
    flare_img : PIL.Image.Image, optional
        Required when ``corruption_name == 'lens_flare'``. A PIL RGB image of
        the lens flare texture.
    illumination_mode : str, optional
        Either ``'shadow'`` or ``'highlight'``; only used for
        ``illumination_variation``. Defaults to ``'shadow'``.

    Returns
    -------
    numpy.ndarray
        The corrupted image (BGR).
    """
    if corruption_name == "camouflage":
        return apply_camouflaging(image, bbox, severity)

    if corruption_name == "dust_scratches":
        return apply_dust_scratches(image, bbox, severity)

    if corruption_name == "fingerprint":
        if fingerprint_texture is None:
            raise ValueError(
                "fingerprint corruption requires a 'fingerprint_texture' image."
            )
        return apply_fingerprint_noise(image, bbox, severity, fingerprint_texture)

    if corruption_name == "focus_shift":
        return apply_object_focus_shift(image, bbox, severity)

    if corruption_name == "illumination_variation":
        return apply_illumination_variation(
            image, bbox, severity, mode=illumination_mode
        )

    if corruption_name == "lens_flare":
        if flare_img is None:
            raise ValueError(
                "lens_flare corruption requires a 'flare_img' PIL image."
            )
        return apply_lens_flare(image, bbox, severity, flare_img)

    if corruption_name == "occlusion":
        return apply_partial_occlusion(image, bbox, severity)

    raise ValueError(
        f"Unknown corruption '{corruption_name}'. "
        f"Supported: {SUPPORTED_CORRUPTIONS}"
    )
