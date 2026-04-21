# ImageNet-LC: Assessing Robustness under Localized Corruptions

Official code release for our paper **"ImageNet-LC: Assessing Robustness
under Localized Corruptions"** (ICPR CO 2026).

ImageNet-LC is an object-centric robustness benchmark. Instead of applying
corruptions globally to every pixel, the pipeline first localizes the
foreground object(s) with a detector and then applies the corruption
**only inside those regions of interest (RoIs)** — the rest of the scene is
untouched. Seven corruption types are supported, each at five severity
levels:

| Corruption                | Simulates                                    |
|---------------------------|----------------------------------------------|
| `lens_flare`              | Lens flare from strong light sources         |
| `illumination_variation`  | Localized brightness/contrast changes        |
| `dust_scratches`          | Dust and surface scratches on the lens       |
| `fingerprint`             | Fingerprint smudges on the lens              |
| `focus_shift`             | Defocus blur on the object region            |
| `occlusion`               | Partial occlusion of the primary object      |
| `camouflage`              | Object blending into the surrounding scene   |

---

## The four-stage pipeline

```
            Stage 1                Stage 2                Stage 3              Stage 4
        ┌──────────┐          ┌──────────────┐        ┌───────────┐       ┌────────────┐
dataset │  YOLOv11 │ labels  │  apply       │ images │ inference │ preds │ evaluation │ report
  ────► │  detect  │ ──────► │  corruptions │ ─────► │  on model │ ────► │  mode      │ ──────►
        └──────────┘          └──────────────┘        └───────────┘       └────────────┘
```

Each stage can be run on its own. Already have labels? Skip Stage 1. Already
have corrupted images? Skip ahead to Stage 3. Already have predictions for
every model? Run only Stage 4.

All stages are accessible through a single entry point (`main.py`),
available in two modes:

- **Interactive menu** (just run `python main.py`)
- **Non-interactive CLI** (use `--stage N` plus the per-stage flags, ideal
  for scripting and cluster jobs)

---

## Repository layout

```
ImageNet-LC/
├── main.py                                # menu + CLI entry point
├── imagenet_lc/
│   ├── stages/
│   │   ├── stage1_bboxes.py               # YOLOv11 -> YOLO labels
│   │   ├── stage2_corrupt.py              # labels -> corrupted images
│   │   ├── stage3_inference.py            # corrupted images -> predictions JSON
│   │   └── stage4_evaluate.py             # predictions -> CE/mCE/LPIPS report
│   ├── corruptions/                       # the 7 localized corruptions
│   ├── models/
│   │   ├── base.py                        # unified classifier interface
│   │   ├── torch_models.py                # torchvision-based CNNs
│   │   └── timm_models.py                 # timm-based (Xception, ViT, DeiT, Swin, ...)
│   ├── metrics/
│   │   ├── corruption_error.py            # CE, mCE, Relative CE, Relative mCE
│   │   └── lpips_metric.py                # LPIPS (paper Table 2)
│   ├── localize.py                        # YOLOv11 wrapper
│   ├── pipeline.py                        # corruption dispatcher
│   ├── io_utils.py                        # label IO, artifact loaders
│   └── data/imagenet_wnids.txt            # canonical 1000 ImageNet WNIDs
├── artifacts/corruptions/                 # texture assets
│   ├── fingerprint.jpg
│   └── flares/*.{png,jpg,jpeg}            # Flares7K patches
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/ImageNet-LC.git
cd ImageNet-LC

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Python 3.8 or newer is required. A CUDA GPU is strongly recommended for
Stage 3 (inference across twelve models × 35 corruption-severity subsets),
but CPU works for smaller subsets.

---

## Dataset preparation (ImageNet)

The pipeline expects the standard ImageNet class-subdirectory layout:

```
imagenet_val/
├── n01440764/                   # synset: tench
│   ├── ILSVRC2012_val_00000293.JPEG
│   ├── ILSVRC2012_val_00002138.JPEG
│   └── ...
├── n01443537/                   # synset: goldfish
│   └── ...
└── ...                          # 1000 synset folders total
```

Steps:

1. Download the ILSVRC2012 validation set (`ILSVRC2012_img_val.tar`) from
   [image-net.org](https://www.image-net.org/) (free registration required).
2. Extract and re-arrange into one-folder-per-class using e.g. the
   [`valprep.sh`](https://github.com/soumith/imagenetloader.torch/blob/master/valprep.sh)
   script.
3. (Optional) To develop or smoke-test, work on a subset — e.g. the first
   20 classes, or 50 images per class. The pipeline handles subsets the
   same way as the full validation set; only inference time changes.

A flat layout (`imagenet_val/<image>.JPEG` with no class subdirs) also
works for quick tests; the `<class>` component in all output trees will
be empty in that case.

---

## Quick start

### Interactive menu

```
$ python main.py

==== ImageNet-LC Pipeline ====
1. Generate bounding boxes (YOLOv11)
2. Apply corruptions
3. Run inference
4. Evaluation mode (CE, mCE, Relative mCE, LPIPS)
q. Quit
Select [1]:
```

Each stage then prompts for paths, subset selection, etc. You can pick
specific corruptions ("all" or a comma-separated list) and specific models.

### End-to-end CLI sequence

```bash
# Stage 1 — one-off: detect and cache YOLOv11 bboxes.
python main.py --stage 1 \
    --dataset /path/to/imagenet_val \
    --labels  /path/to/labels \
    --yolo-weights yolo11n.pt \
    --top-k 3

# Stage 2 — generate the full ImageNet-LC corrupted tree.
python main.py --stage 2 \
    --dataset /path/to/imagenet_val \
    --labels  /path/to/labels \
    --output  /path/to/corrupted \
    --corruptions all \
    --severity 1 2 3 4 5

# Stage 3 — run inference for every model (repeat per model).
for model in alexnet resnet50v2 xception inceptionv3 mobilenetv2 \
             densenet201 vgg19 efficientnetb1 nasnetmobile \
             deit vit swin ; do
    python main.py --stage 3 \
        --data-dir    /path/to/corrupted \
        --mode        corrupted \
        --model       $model \
        --output-file predictions/${model}_corrupted.json
    python main.py --stage 3 \
        --data-dir    /path/to/imagenet_val \
        --mode        clean \
        --model       $model \
        --output-file predictions/${model}_clean.json
done

# Stage 4 — compute every metric the paper reports.
python main.py --stage 4 \
    --predictions-dir predictions \
    --output-dir      eval_output \
    --clean-dir       /path/to/imagenet_val \
    --corrupted-dir   /path/to/corrupted
```

---

## Stage reference

### Stage 1 — Bounding-box generation

Runs YOLOv11 (Ultralytics) over every image in the dataset and writes
YOLO-format `.txt` label files, mirroring the dataset layout. Top-*k* boxes
per image are kept, ranked by detector confidence.

| Flag              | Default        | Description                                              |
|-------------------|----------------|----------------------------------------------------------|
| `--dataset`       | *required*     | Input dataset directory.                                 |
| `--labels`        | *required*     | Output labels directory (mirrors dataset layout).        |
| `--yolo-weights`  | `yolo11n.pt`   | YOLO weights. Auto-downloaded if not present.            |
| `--top-k`         | `3`            | Maximum RoIs per image (paper Section 3.1).              |
| `--skip-existing` | off            | Skip images that already have a label file.             |

Images where YOLO returns no detections get an empty `.txt` file and are
skipped by Stage 2.

### Stage 2 — Corruption application

Reads labels from Stage 1 (or generates random patches for the Section 5.4
ablation) and applies each selected corruption at each severity level
inside every bounding box.

| Flag                      | Default                                  | Description                                                  |
|---------------------------|------------------------------------------|--------------------------------------------------------------|
| `--dataset`               | *required*                               | Input dataset directory.                                     |
| `--labels`                | *required for `--bbox-mode labels`*      | Labels directory from Stage 1.                               |
| `--output`                | *required*                               | Output root for corrupted images.                            |
| `--corruptions`           | *required*                               | `all`, or one or more of the 7 names.                        |
| `--severity`              | `1 2 3 4 5`                              | Severity levels.                                             |
| `--bbox-mode`             | `labels`                                 | `labels` or `random` (Section 5.4 ablation).                 |
| `--num-bboxes`            | `3`                                      | Random bboxes per image (`--bbox-mode random` only).         |
| `--fingerprint-texture`   | `artifacts/corruptions/fingerprint.jpg`  | Texture for the `fingerprint` corruption.                    |
| `--flare-dir`             | `artifacts/corruptions/flares`           | Flares7K patches for `lens_flare`.                            |
| `--illumination-mode`     | `shadow`                                 | `shadow` or `highlight`.                                     |
| `--workers`               | `8`                                      | Thread pool size.                                            |
| `--seed`                  | `None`                                   | RNG seed.                                                    |

**Output layout:** `<output>/<corruption>/<severity>/<class>/<image>`.

### Stage 3 — Inference

Runs one pretrained classifier over the clean dataset or the corrupted
tree and writes per-image top-1 predictions as JSON.

| Flag                | Default                  | Description                                        |
|---------------------|--------------------------|----------------------------------------------------|
| `--data-dir`        | *required*               | Clean dataset (for `--mode clean`) or corrupted tree (for `--mode corrupted`). |
| `--output-file`     | *required*               | Destination JSON path.                             |
| `--model`           | *required*               | One of: see [Supported models](#supported-models). |
| `--mode`            | `corrupted`              | `clean` or `corrupted`.                            |
| `--batch-size`      | `32`                     | Forward-pass batch size.                           |
| `--device`          | auto (cuda if available) | `cpu` or `cuda`.                                   |
| `--wnid-source-dir` | *inferred*               | Override WNID class-folder source.                 |

Each run processes one model. The convention is to write the output as
`predictions/<model>_corrupted.json` and `predictions/<model>_clean.json`
so that Stage 4 can pick them up automatically.

#### Supported models

CNNs: `alexnet`, `resnet50v2`, `vgg19`, `densenet201`, `mobilenetv2`,
`inceptionv3`, `efficientnetb1`, `xception`, `nasnetmobile`.

ViTs: `vit`, `deit`, `swin`.

> **Note on checkpoints.** The paper's numbers were produced using a mix
> of Keras and PyTorch checkpoints (see Muskan's original
> `Inference_Modules`). This repo consolidates everything onto PyTorch
> (torchvision + timm) for simpler dependencies. Absolute accuracies may
> differ slightly; qualitative robustness trends (CNN vs ViT, CE
> rankings, etc.) are preserved.

### Stage 4 — Evaluation mode

Consumes the predictions JSONs from Stage 3 and computes every metric the
paper reports:

- Top-1 error (clean and per `(corruption, severity)`).
- Corruption Error (CE) per corruption, normalised to AlexNet (Eq. 1).
- Mean CE (mCE) — paper Table 1.
- Relative CE and Relative mCE (Eq. 2) — requires `_clean.json` files.
- LPIPS per severity, averaged across corruptions — paper Table 2.

| Flag                      | Default                       | Description                                            |
|---------------------------|-------------------------------|--------------------------------------------------------|
| `--predictions-dir`       | *required*                    | Directory with `<model>_corrupted.json` (+ `_clean.json` optional). |
| `--output-dir`            | *required*                    | Destination for `eval_results.json` and `eval_report.txt`. |
| `--corruptions`           | all 7                         | Corruption types to include in the CE table.           |
| `--clean-dir`             | `None`                        | Clean dataset (required for LPIPS).                    |
| `--corrupted-dir`         | `None`                        | Corrupted tree (required for LPIPS).                   |
| `--no-lpips`              | off                           | Skip LPIPS to save time.                               |
| `--lpips-max-per-class`   | all                           | Cap LPIPS pairs per class.                             |
| `--lpips-backbone`        | `alex`                        | LPIPS backbone (paper uses AlexNet).                   |
| `--baseline-model`        | `alexnet`                     | Model used as the normalisation baseline in Eq. 1/2.   |

Outputs:

- **`eval_results.json`** — machine-readable results for every metric.
- **`eval_report.txt`** — human-readable report (Table 1 + Table 2 style).

---

## Programmatic API

```python
import cv2
from imagenet_lc import apply_corruption, YOLOLocalizer
from imagenet_lc.io_utils import load_fingerprint_texture, load_flare_image

# Phase 1: localize.
localizer = YOLOLocalizer(model_path="yolo11n.pt", top_k=3)
image     = cv2.imread("path/to/image.JPEG")
bboxes    = localizer.detect(image)

# Phase 2: corrupt each RoI.
flare_img = load_flare_image("artifacts/corruptions/flares")
for bbox in bboxes:
    image = apply_corruption(
        "lens_flare", image, bbox, severity=3, flare_img=flare_img,
    )
cv2.imwrite("out.jpg", image)
```

Each stage can also be called as a library:

```python
from imagenet_lc.stages import stage1_bboxes, stage2_corrupt

stage1_bboxes.run(
    dataset_dir="imagenet_val",
    labels_dir="labels",
    top_k=3,
)
stage2_corrupt.run(
    dataset_dir="imagenet_val",
    labels_dir="labels",
    output_dir="corrupted",
    corruptions=["lens_flare", "occlusion"],
    severities=[3, 4, 5],
)
```

---

## Acknowledgments and data credits

- **Flares7K** — the lens-flare patches in `artifacts/corruptions/flares/`
  are derived from the
  [Flares7K dataset](https://github.com/ykdai/Flare7K) (Dai et al., ECCV 2022).
  Please refer to the Flares7K license for downstream use.
- **ImageNet** — ILSVRC2012 (Deng et al., 2009) is the source of all
  input images.
- **Ultralytics** — Stage 1 uses
  [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics).
- **torchvision** and **timm** — Stage 3 pretrained classifiers come from
  [torchvision](https://pytorch.org/vision/) and
  [timm](https://github.com/huggingface/pytorch-image-models).
- **LPIPS** — Stage 4 uses
  [LPIPS](https://github.com/richzhang/PerceptualSimilarity)
  (Zhang et al., CVPR 2018).

---

