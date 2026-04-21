"""
Stage 3: Inference.

Runs a single pretrained classifier over either the clean dataset or a
corrupted dataset tree and saves per-image top-1 predictions to a JSON
file. The evaluation stage consumes these JSON files to compute CE, mCE,
Relative mCE, etc.

Output JSON schema
------------------
{
    "model": "<model_name>",
    "mode": "clean" | "corrupted",
    "predictions": {
        "<corruption>/<severity>/<class_dir>/<image_name>": {
            "true_class_idx": int or null,
            "pred_class_idx": int,
            "correct": bool or null
        },
        ...
    }
}

For clean-dataset runs the key is ``clean/<class_dir>/<image_name>`` so
predictions never collide with corrupted runs.
"""

import json
import os
import sys
from collections import OrderedDict

import torch
from PIL import Image
from tqdm import tqdm

from ..models import load_model, list_models
from ..io_utils import iter_image_files


def _wnid_to_index(dataset_dir):
    """
    Build a mapping from WordNet synset id (class folder name) to the
    integer ImageNet class index used by torchvision/timm pretrained
    models.

    Uses the canonical alphabetical ordering of the 1000 ImageNet-1k
    WNIDs. If the dataset does not use WNID folder names, every class
    is assigned index ``None`` (so ``correct`` will be ``null`` and
    evaluation must be skipped or done against an external ground truth).
    """
    from ..io_utils import IMAGE_EXTENSIONS  # avoid circular

    class_dirs = sorted(
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    )
    # Heuristic: WNIDs are of the form 'n' + 8 digits.
    is_wnid_layout = bool(class_dirs) and all(
        d.startswith("n") and d[1:].isdigit() and len(d) == 9
        for d in class_dirs
    )

    if not is_wnid_layout:
        return {d: None for d in class_dirs}

    # Load the canonical 1000-WNID list. The list ships with torchvision
    # as a meta.bin under the ImageNet weights; for simplicity we embed
    # it at package-data time via a helper file.
    wnids_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "imagenet_wnids.txt"
    )
    wnids_path = os.path.abspath(wnids_path)
    if not os.path.isfile(wnids_path):
        print(
            f"[warn] canonical WNID list not found at {wnids_path}; "
            f"correctness will not be computed.",
            file=sys.stderr,
        )
        return {d: None for d in class_dirs}

    with open(wnids_path) as f:
        canonical = [line.strip() for line in f if line.strip()]
    if len(canonical) != 1000:
        print(
            f"[warn] canonical WNID list has {len(canonical)} entries "
            f"(expected 1000); correctness will not be computed.",
            file=sys.stderr,
        )
        return {d: None for d in class_dirs}

    idx_of = {w: i for i, w in enumerate(canonical)}
    return {d: idx_of.get(d) for d in class_dirs}


def _iter_corrupted_files(corrupted_dir):
    """
    Yield ``(corruption, severity, class_dir, image_name)`` tuples for
    every image under a corrupted-dataset tree with layout:

        corrupted_dir/<corruption>/<severity>/<class_dir>/<image>
    """
    for corruption in sorted(os.listdir(corrupted_dir)):
        c_path = os.path.join(corrupted_dir, corruption)
        if not os.path.isdir(c_path):
            continue
        for severity in sorted(os.listdir(c_path)):
            s_path = os.path.join(c_path, severity)
            if not os.path.isdir(s_path):
                continue
            for class_dir in sorted(os.listdir(s_path)):
                cls_path = os.path.join(s_path, class_dir)
                if not os.path.isdir(cls_path):
                    continue
                for image_name in sorted(os.listdir(cls_path)):
                    yield corruption, severity, class_dir, image_name


def run(
    data_dir,
    output_file,
    model_name,
    mode="corrupted",
    wnid_source_dir=None,
    batch_size=32,
    device=None,
):
    """
    Run inference and dump predictions to ``output_file`` as JSON.

    Parameters
    ----------
    data_dir : str
        Path to the clean dataset (``mode='clean'``) or the corrupted
        dataset tree (``mode='corrupted'``).
    output_file : str
        Destination ``.json`` path. Parent directories are created.
    model_name : str
        One of the registered model short names (see ``models.list_models()``).
    mode : {'clean', 'corrupted'}
        How to interpret ``data_dir``.
    wnid_source_dir : str, optional
        Path to the clean dataset. Used to build the WNID -> class index
        mapping when ``mode='corrupted'``. If omitted, defaults to the
        corrupted tree itself (assuming class folders share names).
    batch_size : int, optional
        Batch size for the forward pass.
    device : str, optional
        Torch device (``'cpu'`` or ``'cuda'``). Defaults to CUDA if available.
    """
    if model_name not in list_models():
        raise SystemExit(
            f"Unknown model '{model_name}'. Registered: {list_models()}"
        )
    if mode not in ("clean", "corrupted"):
        raise SystemExit("mode must be 'clean' or 'corrupted'.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_name} on {device}...")
    classifier = load_model(model_name, device=device)

    # Build WNID -> ImageNet index map.
    wnid_dir = wnid_source_dir or data_dir
    if mode == "corrupted":
        # For corrupted trees we need the class folder layout at the leaves;
        # any (corruption, severity) branch has the right shape.
        probe = None
        for corruption in sorted(os.listdir(wnid_dir)):
            for severity in sorted(os.listdir(os.path.join(wnid_dir, corruption))):
                probe = os.path.join(wnid_dir, corruption, severity)
                break
            if probe:
                break
        if probe is None:
            raise SystemExit(f"No <corruption>/<severity>/<class> tree under {wnid_dir}")
        wnid_map = _wnid_to_index(probe)
    else:
        wnid_map = _wnid_to_index(wnid_dir)

    # Collect inference tasks.
    tasks = []  # list of (key_string, class_dir, image_path)
    if mode == "clean":
        for class_dir, image_name in iter_image_files(data_dir):
            key = f"clean/{class_dir}/{image_name}"
            path = os.path.join(data_dir, class_dir, image_name)
            tasks.append((key, class_dir, path))
    else:
        for corruption, severity, class_dir, image_name in _iter_corrupted_files(
            data_dir
        ):
            key = f"{corruption}/{severity}/{class_dir}/{image_name}"
            path = os.path.join(data_dir, corruption, severity, class_dir, image_name)
            tasks.append((key, class_dir, path))

    if not tasks:
        raise SystemExit(f"No images found under: {data_dir}")

    print(f"Stage 3: running inference on {len(tasks)} images...")
    predictions = OrderedDict()

    # Batched forward pass.
    buffer_keys = []
    buffer_classes = []
    buffer_tensors = []

    def _flush():
        if not buffer_tensors:
            return
        batch = torch.stack(buffer_tensors)
        preds = classifier.predict(batch).cpu().tolist()
        for k, cd, p in zip(buffer_keys, buffer_classes, preds):
            true_idx = wnid_map.get(cd)
            correct = None
            if true_idx is not None:
                correct = bool(p == true_idx)
            predictions[k] = {
                "true_class_idx": true_idx,
                "pred_class_idx": int(p),
                "correct": correct,
            }
        buffer_keys.clear()
        buffer_classes.clear()
        buffer_tensors.clear()

    for key, class_dir, path in tqdm(tasks, desc=f"{model_name}"):
        try:
            img = Image.open(path)
            tensor = classifier.preprocess(img)
        except Exception as e:
            print(f"[warn] could not process {path}: {e}", file=sys.stderr)
            continue
        buffer_keys.append(key)
        buffer_classes.append(class_dir)
        buffer_tensors.append(tensor)
        if len(buffer_tensors) >= batch_size:
            _flush()
    _flush()

    # Write out.
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    payload = {
        "model": model_name,
        "mode": mode,
        "predictions": predictions,
    }
    with open(output_file, "w") as f:
        json.dump(payload, f)

    # Summary.
    scored = [v for v in predictions.values() if v["correct"] is not None]
    if scored:
        top1 = sum(1 for v in scored if v["correct"]) / len(scored)
        print(
            f"Stage 3 complete: {len(predictions)} predictions, "
            f"{len(scored)} scored, top-1 accuracy = {top1:.4f}"
        )
    else:
        print(
            f"Stage 3 complete: {len(predictions)} predictions "
            f"(no WNID mapping, accuracy not computed)."
        )
    print(f"Saved to: {output_file}")
