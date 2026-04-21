import cv2
import numpy as np

def apply_camouflaging(image, bbox, severity):
    try:
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, image.shape[1]), min(y_max, image.shape[0])
        roi = image[y_min:y_max, x_min:x_max]
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[y_min:y_max, x_min:x_max] = 255
        inpainted_image = cv2.inpaint(image, mask[:, :, 0], inpaintRadius=severity * 10, flags=cv2.INPAINT_TELEA)
        blend_ratios = {1: 0.55, 2: 0.67, 3: 0.74, 4: 0.82, 5: 0.90}
        blended_roi = cv2.addWeighted(roi, 1 - blend_ratios[severity], inpainted_image[y_min:y_max, x_min:x_max], blend_ratios[severity], 0)
        image[y_min:y_max, x_min:x_max] = blended_roi
    except Exception as e:
        print(f"Error applying camouflaging: {str(e)}")

    return image
