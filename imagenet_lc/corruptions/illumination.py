import numpy as np

def apply_illumination_variation(image, bbox, severity=3, mode='shadow'):
    """
    Apply a realistic illumination variation (shadow or highlight) centered on a bounding box.
    The effect fades smoothly outside the bounding box.

    Parameters:
        image (np.array): Input BGR image.
        bbox (tuple): (x1, y1, x2, y2) bounding box.
        severity (int): Severity level (1 to 5).
        mode (str): 'shadow' or 'highlight'.

    Returns:
        np.array: Image with variation applied.
    """
    image = image.copy().astype(np.float32)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    box_w, box_h = x2 - x1, y2 - y1

    # Create 2D distance map centered at RoI
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Set max influence radius beyond bbox
    max_radius = int(1.5 * max(box_w, box_h))  # 1.5x bbox size
    fade = np.clip(1 - (dist / max_radius), 0, 1)
    fade = fade ** 2.5  # Sharpen the falloff
    fade = fade[..., None]  # shape: (H, W, 1)

    # Lighting change values
    if mode == 'highlight':
        illum_shift = np.array([20, 20, 35]) * severity  # Cool highlight
    else:
        illum_shift = np.array([-25, -25, -35]) * severity  # Warm shadow

    # Apply radial illumination shift
    shift_layer = fade * illum_shift
    image += shift_layer
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image
