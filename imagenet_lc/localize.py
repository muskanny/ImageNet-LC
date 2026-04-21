"""
Object localization module for ImageNet-LC.

Implements Phase 1 of the ImageNet-LC pipeline (Section 3.1 of the paper):
foreground objects in each image are localized using a YOLO-based object
detector, and the top-k bounding boxes (ranked by confidence) are retained
as regions of interest (RoIs) for subsequent corruption.

The default detector is YOLOv11-nano (``yolo11n.pt``), which matches the
configuration used to generate the results in the paper.
"""

import os

from ultralytics import YOLO


class YOLOLocalizer:
    """
    Thin wrapper around an Ultralytics YOLO detector that returns the
    top-k bounding boxes per image, ranked by confidence.

    Parameters
    ----------
    model_path : str
        Path to the YOLO weights file (e.g. ``yolo11n.pt``). If the file
        does not exist locally, Ultralytics will attempt to download it
        from the official release assets on first use.
    top_k : int, optional
        Maximum number of bounding boxes to keep per image, ranked by
        detection confidence. Defaults to 3 (per paper Section 3.1).
    verbose : bool, optional
        If ``False``, suppress per-inference logging from Ultralytics.
        Defaults to ``False``.
    """

    def __init__(self, model_path="yolo11n.pt", top_k=3, verbose=False):
        self.model_path = model_path
        self.top_k = top_k
        self.verbose = verbose
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Run the detector on a single image and return the top-k bboxes.

        Parameters
        ----------
        image : numpy.ndarray
            Input image in BGR format (as read by ``cv2.imread``).

        Returns
        -------
        list of tuple
            Up to ``top_k`` bounding boxes, each as an integer tuple
            ``(x_min, y_min, x_max, y_max)``, sorted in descending order
            of confidence. Returns an empty list if no objects are detected.
        """
        results = self.model(image, verbose=self.verbose)

        detections = []
        for result in results:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0]
                confidence = float(box.conf[0])
                detections.append(
                    (
                        (
                            int(x_min),
                            int(y_min),
                            int(x_max),
                            int(y_max),
                        ),
                        confidence,
                    )
                )

        # Rank by confidence (descending) and keep the top-k.
        detections.sort(key=lambda d: d[1], reverse=True)
        detections = detections[: self.top_k]

        return [bbox for bbox, _ in detections]


def save_yolo_labels(label_path, bboxes, image_width, image_height, class_ids=None):
    """
    Persist detected bounding boxes in YOLO-normalized .txt format.

    Useful for caching detector outputs so the pipeline can be re-run
    without invoking the detector again.

    Parameters
    ----------
    label_path : str
        Destination .txt path. Parent directories are created if missing.
    bboxes : list of tuple
        Bounding boxes as ``(x_min, y_min, x_max, y_max)`` in absolute pixels.
    image_width, image_height : int
        Image dimensions in pixels.
    class_ids : list of int, optional
        Class id for each bbox. If ``None``, every box is written with
        class id ``0`` (the pipeline does not use class ids for corruption).
    """
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    if class_ids is None:
        class_ids = [0] * len(bboxes)

    with open(label_path, "w") as f:
        for (x_min, y_min, x_max, y_max), class_id in zip(bboxes, class_ids):
            cx = (x_min + x_max) / (2 * image_width)
            cy = (y_min + y_max) / (2 * image_height)
            w = (x_max - x_min) / image_width
            h = (y_max - y_min) / image_height
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
