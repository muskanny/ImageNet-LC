import cv2
import numpy as np

def apply_dust_scratches(image, bbox, severity):
    try:
        x_min, y_min, x_max, y_max = bbox
        corrupted_image = image.copy()
        num_scratches = severity * 3
        num_dust_particles = severity * 20
        for _ in range(num_scratches):
            x_start = np.random.randint(x_min, x_max)
            y_start = np.random.randint(y_min, y_max)
            length = np.random.randint(10, 20 * severity)
            angle = np.random.uniform(0, 2 * np.pi)
            x_end = int(x_start + length * np.cos(angle))
            y_end = int(y_start + length * np.sin(angle))
            scratch_color = (np.random.randint(200, 256), np.random.randint(200, 256), np.random.randint(200, 256))
            cv2.line(corrupted_image, (x_start, y_start), (x_end, y_end), scratch_color, 1)
        for _ in range(num_dust_particles):
            x = np.random.randint(x_min, x_max)
            y = np.random.randint(y_min, y_max)
            dust_color = (np.random.randint(200, 256), np.random.randint(200, 256), np.random.randint(200, 256))
            cv2.circle(corrupted_image, (x, y), 1, dust_color, -1)
        image[y_min:y_max, x_min:x_max] = corrupted_image[y_min:y_max, x_min:x_max]
    except Exception as e:
        print(f"Error applying dust and scratches: {str(e)}")

    return image
