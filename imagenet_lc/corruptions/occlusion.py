import numpy as np

def apply_partial_occlusion(image, bbox, severity):
    try:
        x_min, y_min, x_max, y_max = bbox
        for _ in range(severity):
            occlusion_x = np.random.randint(x_min, x_max)
            occlusion_y = np.random.randint(y_min, y_max)
            patch_width = int(0.2 * (x_max - x_min) * severity / 5)
            patch_height = int(0.2 * (y_max - y_min) * severity / 5)
            color = tuple(np.random.randint(0, 256, size=3))
            image[occlusion_y:occlusion_y+patch_height, occlusion_x:occlusion_x+patch_width] = color
    except Exception as e:
        print(f"Error applying partial occlusion: {str(e)}")

    return image
