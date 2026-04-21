"""
LPIPS metric for ImageNet-LC.

Computes the Learned Perceptual Image Patch Similarity (Zhang et al.,
CVPR 2018) between each corrupted image and its clean counterpart. The
paper reports a single LPIPS number per severity level (averaged across
all seven corruptions, see Table 2).

The implementation follows Muskan's ``lpips_evaluation.py`` in spirit:
AlexNet-backbone LPIPS, images resized to 224x224 and ImageNet-normalised,
per-corruption scores averaged per severity, and finally averaged across
corruptions.
"""

import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# Lazy module-level cache so we don't re-instantiate LPIPS every call.
_LPIPS_MODEL_CACHE = {}


def _get_lpips_model(net="alex", device="cpu"):
    key = (net, device)
    if key not in _LPIPS_MODEL_CACHE:
        try:
            import lpips
        except ImportError as e:
            raise SystemExit(
                "The 'lpips' package is required for LPIPS evaluation. "
                "Install it with `pip install lpips`."
            ) from e
        _LPIPS_MODEL_CACHE[key] = lpips.LPIPS(net=net).to(device).eval()
    return _LPIPS_MODEL_CACHE[key]


_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def _load_tensor(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[warn] LPIPS skip {image_path}: {e}", file=sys.stderr)
        return None
    tensor = _TRANSFORM(img).unsqueeze(0)
    return tensor


def compute(
    clean_dir,
    corrupted_dir,
    net="alex",
    device=None,
    max_images_per_class=None,
    severity_levels=(1, 2, 3, 4, 5),
    corruption_filter=None,
):
    """
    Compute per-severity LPIPS scores averaged across all corruptions.

    Parameters
    ----------
    clean_dir : str
        Clean ImageNet-style dataset root (``clean_dir/<class>/<image>``).
    corrupted_dir : str
        Corrupted tree root (``corrupted_dir/<corruption>/<severity>/<class>/<image>``).
    net : str, optional
        LPIPS backbone (``'alex'`` matches the paper / Muskan's script).
    device : str, optional
        Torch device; defaults to CUDA if available.
    max_images_per_class : int, optional
        If set, only the first N images per class (sorted) are scored.
        Useful for sanity checks on a subset.
    severity_levels : iterable of int
        Severity levels to include.
    corruption_filter : list of str, optional
        If set, only these corruption names are scored.

    Returns
    -------
    dict
        ``{
            'per_severity': {severity: avg_LPIPS_across_corruptions},
            'per_corruption_per_severity': {corr: {sev: avg_LPIPS}},
            'backbone': net,
            'n_pairs': int,
        }``
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _get_lpips_model(net=net, device=device)

    if not os.path.isdir(clean_dir):
        raise SystemExit(f"Clean directory does not exist: {clean_dir}")
    if not os.path.isdir(corrupted_dir):
        raise SystemExit(f"Corrupted directory does not exist: {corrupted_dir}")

    per_corr_sev = {}  # {corruption: {severity: [scores...]}}
    n_pairs = 0

    corruptions = sorted(
        d
        for d in os.listdir(corrupted_dir)
        if os.path.isdir(os.path.join(corrupted_dir, d))
    )
    if corruption_filter is not None:
        corruptions = [c for c in corruptions if c in corruption_filter]

    for corruption in corruptions:
        per_corr_sev.setdefault(corruption, {})
        for sev in severity_levels:
            sev_dir = os.path.join(corrupted_dir, corruption, str(sev))
            if not os.path.isdir(sev_dir):
                continue

            classes = sorted(
                d
                for d in os.listdir(sev_dir)
                if os.path.isdir(os.path.join(sev_dir, d))
            )
            scores = []

            for class_name in classes:
                class_sev_dir = os.path.join(sev_dir, class_name)
                images = sorted(os.listdir(class_sev_dir))
                if max_images_per_class is not None:
                    images = images[:max_images_per_class]

                for img_name in tqdm(
                    images,
                    desc=f"{corruption}/sev{sev}/{class_name}",
                    leave=False,
                ):
                    corr_path = os.path.join(class_sev_dir, img_name)
                    clean_path = os.path.join(clean_dir, class_name, img_name)
                    if not os.path.isfile(clean_path):
                        continue

                    t_corr = _load_tensor(corr_path)
                    t_clean = _load_tensor(clean_path)
                    if t_corr is None or t_clean is None:
                        continue

                    with torch.no_grad():
                        score = model.forward(
                            t_clean.to(device), t_corr.to(device)
                        ).item()
                    if np.isnan(score) or np.isinf(score):
                        continue
                    scores.append(score)
                    n_pairs += 1

            if scores:
                per_corr_sev[corruption][sev] = float(np.mean(scores))

    # Average across corruptions for each severity.
    per_severity = {}
    for sev in severity_levels:
        vals = [
            per_corr_sev[c][sev]
            for c in per_corr_sev
            if sev in per_corr_sev[c]
        ]
        if vals:
            per_severity[sev] = float(np.mean(vals))

    return {
        "per_severity": per_severity,
        "per_corruption_per_severity": per_corr_sev,
        "backbone": net,
        "n_pairs": n_pairs,
    }


def format_severity_table(per_severity_dict):
    """Format the Table 2 summary."""
    lines = ["Severity | LPIPS", "-" * 20]
    for sev in sorted(per_severity_dict):
        lines.append(f"   {sev}     | {per_severity_dict[sev]:.4f}")
    return "\n".join(lines)
