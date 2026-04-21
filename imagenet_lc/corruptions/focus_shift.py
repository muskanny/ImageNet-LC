import cv2

def apply_object_focus_shift(image, bbox, severity):
    try:
        x_min, y_min, x_max, y_max = bbox
        cropped = image[y_min:y_max, x_min:x_max]
        blurred = cv2.GaussianBlur(cropped, (severity * 2 + 1, severity * 2 + 1), severity)
        image[y_min:y_max, x_min:x_max] = blurred
    except Exception as e:
        print(f"Error applying object focus shift: {str(e)}")

    return image
