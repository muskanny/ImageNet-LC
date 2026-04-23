import numpy as np
import albumentations as A

class CustomOcclusion(A.ImageOnlyTransform):
    
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, bbox=None, severity=1, **params):
        if bbox is None:
            return img

        # Pipeline provides absolute pixel coordinates: (x_min, y_min, x_max, y_max)
        h, w = img.shape[:2]
        x_min, y_min, x_max, y_max = map(int, bbox)
        
        # Clip coordinates to image boundaries
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)
        
        obj_w = x_max - x_min
        obj_h = y_max - y_min

        if obj_w <= 0 or obj_h <= 0:
            return img

        # Calculate patch dimensions based on severity (10% to 50% of object size)
        # S1 = 10% of object width/height, S5 = 50% of object width/height
        p_w = int(obj_w * severity * 0.1)
        p_h = int(obj_h * severity * 0.1)
        
        # Ensure the patch is at least 1x1 pixel
        p_w, p_h = max(1, p_w), max(1, p_h)

        # Randomly select a top-left starting point that keeps the patch INSIDE the bbox
        # Range: [x_min, x_max - p_w]. Use max to handle cases where p_w == obj_w.
        start_x = np.random.randint(x_min, max(x_min + 1, x_max - p_w + 1))
        start_y = np.random.randint(y_min, max(y_min + 1, y_max - p_h + 1))

        # Apply occlusion
        occluded_img = img.copy()
        
        # Calculate end coordinates and ensure they don't exceed image bounds
        end_x = min(start_x + p_w, w)
        end_y = min(start_y + p_h, h)

        # Apply black patch (matches Figure 2 in paper)
        occluded_img[start_y:end_y, start_x:end_x, :] = 0 

        return occluded_img

def apply_partial_occlusion(image, bbox, severity):
    """
    Main entry point for the ImageNet-LC Stage 2 pipeline.
    """
    transform = CustomOcclusion()
    return transform.apply(image, bbox=bbox, severity=severity)