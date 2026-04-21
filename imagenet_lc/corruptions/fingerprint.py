import cv2
import numpy as np

def apply_fingerprint_noise(image, bbox, severity, fingerprint_texture):
    try:
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(image.shape[1], x_max), min(image.shape[0], y_max)

        roi = image[y_min:y_max, x_min:x_max]
        h, w = roi.shape[:2]

        # Resize fingerprint to match the RoI
        fingerprint_resized = cv2.resize(fingerprint_texture, (w, h))

        # Convert to grayscale and threshold to extract fingerprint (black = foreground)
        gray_fp = cv2.cvtColor(fingerprint_resized, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_fp, 220, 255, cv2.THRESH_BINARY_INV)  # white bg → remove

        # Optional: blur mask to simulate smudge
        binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)

        # Normalize alpha mask and scale with severity
        alpha = (binary_mask.astype(np.float32) / 255.0) * np.clip(severity * 0.3, 0.2, 0.9)
        alpha = alpha[..., None]  # (H, W, 1)

        # Use grayscale fingerprint texture for color (optional: you can tint or keep gray)
        fingerprint_color = cv2.cvtColor(gray_fp, cv2.COLOR_GRAY2BGR).astype(np.float32)
        roi_float = roi.astype(np.float32)

        # Blend only fingerprint ridges
        blended = (1 - alpha) * roi_float + alpha * fingerprint_color
        image[y_min:y_max, x_min:x_max] = np.clip(blended, 0, 255).astype(np.uint8)

    except Exception as e:
        print(f"Error applying fingerprint noise: {str(e)}")

    return image
