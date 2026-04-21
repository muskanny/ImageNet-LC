import cv2
import numpy as np

from PIL import Image

def apply_lens_flare(image, bbox, severity=3, flare_img=None):
    try:
        severity = int(np.clip(severity, 1, 5))

        x1, y1, x2, y2 = bbox
        bbox_w, bbox_h = x2 - x1, y2 - y1

        # Severity affects size and opacity
        size_factor = 1.0 + 0.3 * severity
        base_alpha = 0.3 + 0.15 * severity  # Controls max opacity

        flare_w = int(bbox_w * size_factor)
        flare_h = int(bbox_h * size_factor)

        flare_patch = flare_img.resize((flare_w, flare_h), Image.Resampling.LANCZOS)
        flare_patch_np = np.asarray(flare_patch)
        flare_patch_bgr = cv2.cvtColor(flare_patch_np, cv2.COLOR_RGB2BGR)

        # Convert to grayscale to get brightness mask
        flare_gray = cv2.cvtColor(flare_patch_bgr, cv2.COLOR_BGR2GRAY)
        brightness = flare_gray.astype(np.float32) / 255.0
        alpha_mask = np.clip(brightness * base_alpha, 0, 1)

        # Determine center point inside bbox
        cx = x1 + bbox_w // 2
        cy = y1 + bbox_h // 2

        # With jitter
        # jitter = max(1, severity)  # small randomness
        # cx = x1 + bbox_w // 2 + random.randint(-jitter, jitter)
        # cy = y1 + bbox_h // 2 + random.randint(-jitter, jitter)

        # Compute flare position
        x_start = max(cx - flare_w // 2, 0)
        y_start = max(cy - flare_h // 2, 0)
        x_end = min(x_start + flare_w, image.shape[1])
        y_end = min(y_start + flare_h, image.shape[0])

        # Resize flare if clipped
        flare_patch_bgr = flare_patch_bgr[:y_end - y_start, :x_end - x_start]
        alpha_mask = alpha_mask[:y_end - y_start, :x_end - x_start]

        # Prepare overlay and apply
        overlay = np.zeros_like(image)
        overlay[y_start:y_end, x_start:x_end] = flare_patch_bgr

        alpha_mask = alpha_mask[..., None]  # Make shape (H, W, 1)
        image[y_start:y_end, x_start:x_end] = (
            (1 - alpha_mask) * image[y_start:y_end, x_start:x_end] +
            alpha_mask * overlay[y_start:y_end, x_start:x_end]
        ).astype(np.uint8)

        # Draw bbox for visualization
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    except Exception as e:
        print(f"Error applying lens flare: {str(e)}")

    return image
