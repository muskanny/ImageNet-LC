"""
I/O utilities for ImageNet-LC.

- ``read_yolo_bboxes``: read a YOLO-format .txt label file and convert to
  absolute (x_min, y_min, x_max, y_max) pixel coordinates.
- ``generate_random_bboxes``: generate N random bounding boxes within an
  image, useful when no labels are provided.
- ``load_fingerprint_texture`` / ``load_flare_image``: load the artifact
  textures used by the fingerprint and lens_flare corruptions.
"""

import glob
import os
import random

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")


def read_yolo_bboxes(label_file, image_width, image_height):
    """
    Read a YOLO-format .txt label file and return absolute bounding boxes.

    YOLO format per line: ``class x_center y_center width height`` (all
    values for coordinates are normalized to [0, 1]).

    Parameters
    ----------
    label_file : str
        Path to the YOLO .txt label file.
    image_width : int
        Width of the corresponding image, in pixels.
    image_height : int
        Height of the corresponding image, in pixels.

    Returns
    -------
    list of tuple
        A list of ``(x_min, y_min, x_max, y_max)`` bounding boxes in pixel
        coordinates.
    """
    bboxes = []
    with open(label_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            _, x_center, y_center, width, height = map(float, parts)
            x_min = int((x_center - width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            x_max = int((x_center + width / 2) * image_width)
            y_max = int((y_center + height / 2) * image_height)
            bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes


def generate_random_bboxes(image_width, image_height, num_boxes, seed=None):
    """
    Generate ``num_boxes`` random bounding boxes within an image.

    Each box covers between 20% and 50% of the image in each dimension.
    Useful when no ground-truth labels are available.

    Parameters
    ----------
    image_width, image_height : int
        Image dimensions in pixels.
    num_boxes : int
        Number of bounding boxes to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list of tuple
        A list of ``(x_min, y_min, x_max, y_max)`` bounding boxes.
    """
    rng = random.Random(seed)
    bboxes = []
    for _ in range(num_boxes):
        box_w = rng.randint(int(0.2 * image_width), int(0.5 * image_width))
        box_h = rng.randint(int(0.2 * image_height), int(0.5 * image_height))
        x_min = rng.randint(0, max(image_width - box_w, 1))
        y_min = rng.randint(0, max(image_height - box_h, 1))
        bboxes.append((x_min, y_min, x_min + box_w, y_min + box_h))
    return bboxes


def load_fingerprint_texture(path):
    """Load the fingerprint texture as a BGR ``numpy.ndarray``."""
    texture = cv2.imread(path)
    if texture is None:
        raise FileNotFoundError(f"Could not read fingerprint texture at: {path}")
    return texture


def load_flare_image(flare_dir, seed=None):
    """
    Pick and load a random lens-flare image from a directory.

    Parameters
    ----------
    flare_dir : str
        Directory containing flare images (png/jpg/jpeg).
    seed : int, optional
        Random seed for reproducible flare selection.

    Returns
    -------
    PIL.Image.Image
        The chosen flare as a PIL RGB image.
    """
    flare_paths = []
    for ext in ("png", "jpg", "jpeg"):
        flare_paths.extend(glob.glob(os.path.join(flare_dir, f"*.{ext}")))
    if not flare_paths:
        raise ValueError(f"No flare images found in: {flare_dir}")
    rng = random.Random(seed)
    return Image.open(rng.choice(flare_paths)).convert("RGB")


def iter_image_files(root):
    """
    Yield ``(class_dir, image_name)`` pairs for every image under ``root``.

    Supports two layouts:
      1. Class-subdirectory layout: ``root/<class>/<image>``.
      2. Flat layout: ``root/<image>`` (yields ``class_dir == ''``).

    Hidden files and non-image files are skipped.
    """
    # Check for class subdirectory layout.
    entries = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    ]
    if entries:
        for class_dir in entries:
            class_path = os.path.join(root, class_dir)
            for image_name in os.listdir(class_path):
                if image_name.startswith("."):
                    continue
                if image_name.lower().endswith(IMAGE_EXTENSIONS):
                    yield class_dir, image_name
    else:
        # Flat layout.
        for image_name in os.listdir(root):
            if image_name.startswith("."):
                continue
            if image_name.lower().endswith(IMAGE_EXTENSIONS):
                yield "", image_name
