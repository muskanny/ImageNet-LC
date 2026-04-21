"""
Stage 1: Bounding box generation.

Runs a YOLO-based object detector over every image in the dataset and
writes the top-k bounding boxes (ranked by confidence) as YOLO-format
``.txt`` label files that mirror the dataset directory layout. This is
Phase 1 of the pipeline described in Section 3 of the paper.
"""

import os
import sys

import cv2
from tqdm import tqdm

from ..localize import YOLOLocalizer, save_yolo_labels
from ..io_utils import iter_image_files


def run(
    dataset_dir,
    labels_dir,
    yolo_weights="yolo11n.pt",
    top_k=3,
    skip_existing=False,
):
    """
    Generate YOLO-format bounding box labels for every image in
    ``dataset_dir`` and write them under ``labels_dir``.

    Parameters
    ----------
    dataset_dir : str
        Input dataset directory (class-subdirectory or flat layout).
    labels_dir : str
        Output labels directory. Mirrors the input layout. Created if it
        does not exist.
    yolo_weights : str, optional
        Path to YOLO weights. Defaults to ``yolo11n.pt`` (auto-downloaded
        by Ultralytics on first use).
    top_k : int, optional
        Maximum number of RoIs per image, sorted by confidence. Defaults
        to 3 (paper setting).
    skip_existing : bool, optional
        If True, images for which a label file already exists are skipped.
        Useful for resuming an interrupted run.

    Returns
    -------
    dict
        Summary with keys ``total``, ``detected``, ``skipped``, ``failed``.
    """
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
    os.makedirs(labels_dir, exist_ok=True)

    all_files = list(iter_image_files(dataset_dir))
    if not all_files:
        raise SystemExit(f"No images found under: {dataset_dir}")

    print(
        f"Stage 1: running YOLO ({yolo_weights}, top_k={top_k}) "
        f"on {len(all_files)} images."
    )
    localizer = YOLOLocalizer(
        model_path=yolo_weights, top_k=top_k, verbose=False
    )

    stats = {"total": len(all_files), "detected": 0, "skipped": 0, "failed": 0}

    for class_dir, image_name in tqdm(all_files, desc="detecting"):
        label_path = os.path.join(
            labels_dir, class_dir, os.path.splitext(image_name)[0] + ".txt"
        )

        if skip_existing and os.path.isfile(label_path):
            stats["skipped"] += 1
            continue

        image_path = os.path.join(dataset_dir, class_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(
                f"[warn] could not read: {image_path}", file=sys.stderr
            )
            stats["failed"] += 1
            continue

        bboxes = localizer.detect(image)
        if not bboxes:
            # No detections: still write an empty file so downstream stages
            # can tell "we tried and found nothing" apart from "never ran".
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            open(label_path, "w").close()
            continue

        h, w = image.shape[:2]
        save_yolo_labels(label_path, bboxes, w, h)
        stats["detected"] += 1

    print(
        f"Stage 1 complete: detected in {stats['detected']} / {stats['total']} "
        f"images, skipped {stats['skipped']}, failed {stats['failed']}."
    )
    return stats
