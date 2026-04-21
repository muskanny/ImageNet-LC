"""
Stage 2: Corruption application.

For every image in the dataset, reads the YOLO bounding boxes from the
labels directory and applies one or more corruptions at one or more
severity levels within each bbox. Outputs are written to:

    <output_dir>/<corruption>/<severity>/<class_dir>/<image_name>
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from tqdm import tqdm

from ..pipeline import apply_corruption, SUPPORTED_CORRUPTIONS
from ..io_utils import (
    read_yolo_bboxes,
    generate_random_bboxes,
    load_fingerprint_texture,
    load_flare_image,
    iter_image_files,
)


def _process_one(
    image_name,
    class_dir,
    dataset_dir,
    output_dir,
    corruption,
    severity,
    bboxes,
    fingerprint_texture,
    flare_img,
    illumination_mode,
):
    image_path = os.path.join(dataset_dir, class_dir, image_name)
    image = cv2.imread(image_path)
    if image is None:
        return

    corrupted = image.copy()
    for bbox in bboxes:
        corrupted = apply_corruption(
            corruption,
            corrupted,
            bbox,
            severity,
            fingerprint_texture=fingerprint_texture,
            flare_img=flare_img,
            illumination_mode=illumination_mode,
        )

    out_subdir = os.path.join(output_dir, corruption, str(severity), class_dir)
    os.makedirs(out_subdir, exist_ok=True)
    cv2.imwrite(os.path.join(out_subdir, image_name), corrupted)


def run(
    dataset_dir,
    labels_dir,
    output_dir,
    corruptions,
    severities=(1, 2, 3, 4, 5),
    bbox_mode="labels",
    num_random_bboxes=3,
    fingerprint_texture_path="artifacts/corruptions/fingerprint.jpg",
    flare_dir="artifacts/corruptions/flares",
    illumination_mode="shadow",
    workers=8,
    seed=None,
):
    """
    Apply corruptions within bounding-box regions for an entire dataset.

    Parameters
    ----------
    dataset_dir : str
        Input dataset directory.
    labels_dir : str or None
        YOLO labels directory (required when ``bbox_mode == 'labels'``).
        Ignored otherwise.
    output_dir : str
        Destination for corrupted images.
    corruptions : list of str
        Corruption names to apply. Each must be in ``SUPPORTED_CORRUPTIONS``.
    severities : iterable of int, optional
        Severity levels in ``[1, 5]``. Defaults to all five.
    bbox_mode : {'labels', 'random'}, optional
        Where bboxes come from. ``'labels'`` reads YOLO ``.txt`` files
        from ``labels_dir``; ``'random'`` generates random patches (used
        for the Section 5.4 ablation).
    num_random_bboxes : int, optional
        Number of random bboxes per image when ``bbox_mode == 'random'``.
    fingerprint_texture_path, flare_dir : str, optional
        Paths to corruption assets.
    illumination_mode : {'shadow', 'highlight'}, optional
        Used by ``illumination_variation``.
    workers : int, optional
        Thread pool size.
    seed : int, optional
        RNG seed for reproducible random bboxes and flare selection.
    """
    unknown = [c for c in corruptions if c not in SUPPORTED_CORRUPTIONS]
    if unknown:
        raise SystemExit(
            f"Unknown corruption(s): {unknown}. "
            f"Supported: {SUPPORTED_CORRUPTIONS}"
        )
    for s in severities:
        if s < 1 or s > 5:
            raise SystemExit(f"Severity must be in [1, 5], got {s}.")
    if bbox_mode not in ("labels", "random"):
        raise SystemExit(f"bbox_mode must be 'labels' or 'random'.")
    if bbox_mode == "labels" and not labels_dir:
        raise SystemExit("bbox_mode='labels' requires a labels_dir.")

    os.makedirs(output_dir, exist_ok=True)

    # Lazy-load corruption assets only if needed.
    fingerprint_texture = None
    if "fingerprint" in corruptions:
        fingerprint_texture = load_fingerprint_texture(fingerprint_texture_path)

    flare_img = None
    if "lens_flare" in corruptions:
        flare_img = load_flare_image(flare_dir, seed=seed)

    all_files = list(iter_image_files(dataset_dir))
    if not all_files:
        raise SystemExit(f"No images found under: {dataset_dir}")

    # Resolve bboxes per image.
    bbox_map = {}
    for class_dir, image_name in all_files:
        image_path = os.path.join(dataset_dir, class_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]

        if bbox_mode == "labels":
            label_file = os.path.join(
                labels_dir,
                class_dir,
                os.path.splitext(image_name)[0] + ".txt",
            )
            if not os.path.isfile(label_file):
                print(
                    f"[warn] missing label file: {label_file}",
                    file=sys.stderr,
                )
                continue
            bboxes = read_yolo_bboxes(label_file, w, h)
            if not bboxes:
                # Empty label file means "detector found nothing". Skip.
                continue
        else:
            bboxes = generate_random_bboxes(w, h, num_random_bboxes, seed=seed)

        bbox_map[(class_dir, image_name)] = bboxes

    phase2_files = [k for k in all_files if k in bbox_map]
    print(
        f"Stage 2: applying {len(corruptions)} corruption(s) x "
        f"{len(severities)} severity level(s) to {len(phase2_files)} images."
    )

    for corruption in corruptions:
        for severity in severities:
            desc = f"{corruption} @ severity {severity}"
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _process_one,
                        image_name,
                        class_dir,
                        dataset_dir,
                        output_dir,
                        corruption,
                        severity,
                        bbox_map[(class_dir, image_name)],
                        fingerprint_texture,
                        flare_img,
                        illumination_mode,
                    )
                    for class_dir, image_name in phase2_files
                ]
                for _ in tqdm(
                    as_completed(futures), total=len(futures), desc=desc
                ):
                    pass

    print(f"Stage 2 complete. Output: {output_dir}")
