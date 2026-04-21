"""
ImageNet-LC: menu-driven pipeline entry point.

The full pipeline has four independent stages. Each can be entered on its
own, so you can resume from any point if you already have artefacts from
a previous run.

    Stage 1 : Bounding-box generation   (YOLOv11 -> YOLO .txt labels)
    Stage 2 : Corruption application     (image tree -> corrupted tree)
    Stage 3 : Inference                  (one model -> predictions JSON)
    Stage 4 : Evaluation                 (all metrics: CE, mCE, LPIPS, ...)

Invocation
----------
Run with no arguments to get the interactive menu:

    python main.py

Or select a stage non-interactively via ``--stage {1,2,3,4}``:

    python main.py --stage 1 --dataset ... --labels ...
    python main.py --stage 2 --dataset ... --labels ... --output ... --corruptions all
    python main.py --stage 3 --data-dir ... --model resnet50v2 --output-file ...
    python main.py --stage 4 --predictions-dir ... --output-dir ...

Each stage has its own flags; run ``python main.py --stage <N> --help`` for
the per-stage reference.
"""

import argparse
import os
import sys

from imagenet_lc.pipeline import SUPPORTED_CORRUPTIONS
from imagenet_lc.models import list_models


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def _prompt(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default


def _prompt_int(prompt, default=None, min_val=None, max_val=None):
    while True:
        raw = _prompt(prompt, default=str(default) if default is not None else None)
        try:
            val = int(raw)
        except ValueError:
            print("  (not an integer, try again)")
            continue
        if min_val is not None and val < min_val:
            print(f"  (must be >= {min_val})")
            continue
        if max_val is not None and val > max_val:
            print(f"  (must be <= {max_val})")
            continue
        return val


def _prompt_choice(prompt, choices, default=None):
    """Let the user pick a single option by number or name."""
    print(prompt)
    for i, c in enumerate(choices, 1):
        marker = " *" if c == default else ""
        print(f"  {i}. {c}{marker}")
    while True:
        raw = _prompt("Choose", default=default or "1")
        if raw in choices:
            return raw
        try:
            i = int(raw)
        except ValueError:
            print("  (not a valid choice)")
            continue
        if 1 <= i <= len(choices):
            return choices[i - 1]
        print(f"  (choose 1..{len(choices)} or type a name)")


def _prompt_multi(prompt, choices, default_all=True):
    """Multi-pick: 'all' / comma-separated list."""
    print(prompt)
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    default = "all" if default_all else ""
    raw = _prompt("Enter 'all' or comma-separated names/numbers", default=default)
    if raw.lower() == "all":
        return list(choices)
    picks = [p.strip() for p in raw.split(",") if p.strip()]
    out = []
    for p in picks:
        if p in choices:
            out.append(p)
            continue
        try:
            i = int(p)
        except ValueError:
            print(f"  (skipping unknown: {p})")
            continue
        if 1 <= i <= len(choices):
            out.append(choices[i - 1])
        else:
            print(f"  (skipping out-of-range: {p})")
    if not out:
        raise SystemExit("No valid selections.")
    return out


def _prompt_yesno(prompt, default=True):
    default_str = "y" if default else "n"
    while True:
        raw = _prompt(prompt + " (y/n)", default=default_str).lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


# ---------------------------------------------------------------------------
# Interactive wrappers around each stage
# ---------------------------------------------------------------------------

def _run_stage1_interactive():
    from imagenet_lc.stages import stage1_bboxes

    print("\n== Stage 1: Bounding-box generation (YOLOv11) ==\n")
    dataset = _prompt("Path to dataset directory")
    labels = _prompt("Path to output labels directory")
    weights = _prompt("YOLO weights", default="yolo11n.pt")
    top_k = _prompt_int("Top-k boxes per image", default=3, min_val=1)
    skip = _prompt_yesno("Skip images that already have a label file?", default=False)
    stage1_bboxes.run(
        dataset_dir=dataset,
        labels_dir=labels,
        yolo_weights=weights,
        top_k=top_k,
        skip_existing=skip,
    )


def _run_stage2_interactive():
    from imagenet_lc.stages import stage2_corrupt

    print("\n== Stage 2: Corruption application ==\n")
    dataset = _prompt("Path to dataset directory")
    bbox_mode = _prompt_choice(
        "Bounding-box source:", ["labels", "random"], default="labels"
    )
    labels = None
    num_rand = 3
    if bbox_mode == "labels":
        labels = _prompt("Path to YOLO labels directory")
    else:
        num_rand = _prompt_int("Random boxes per image", default=3, min_val=1)

    output = _prompt("Path to output directory for corrupted images")
    corruptions = _prompt_multi(
        "Which corruptions to apply?", SUPPORTED_CORRUPTIONS, default_all=True
    )
    sev_raw = _prompt(
        "Severity levels to apply (space-separated, 1-5)", default="1 2 3 4 5"
    )
    severities = [int(x) for x in sev_raw.split()]
    workers = _prompt_int("Worker threads", default=8, min_val=1)
    stage2_corrupt.run(
        dataset_dir=dataset,
        labels_dir=labels,
        output_dir=output,
        corruptions=corruptions,
        severities=severities,
        bbox_mode=bbox_mode,
        num_random_bboxes=num_rand,
        workers=workers,
    )


def _run_stage3_interactive():
    from imagenet_lc.stages import stage3_inference

    print("\n== Stage 3: Inference ==\n")
    mode = _prompt_choice(
        "Inference on corrupted tree or clean dataset?",
        ["corrupted", "clean"],
        default="corrupted",
    )
    data_dir = _prompt(
        "Path to " + ("corrupted tree" if mode == "corrupted" else "clean dataset")
    )

    models = _prompt_multi(
        "Which model(s) to run?", list_models(), default_all=False
    )

    output_dir = _prompt("Output directory for predictions JSONs", default="predictions")
    batch = _prompt_int("Batch size", default=32, min_val=1)
    device = _prompt("Device (cpu / cuda)", default="cpu")

    os.makedirs(output_dir, exist_ok=True)
    for model in models:
        out_file = os.path.join(output_dir, f"{model}_{mode}.json")
        print(f"\n--- Running {model} ({mode}) ---")
        stage3_inference.run(
            data_dir=data_dir,
            output_file=out_file,
            model_name=model,
            mode=mode,
            batch_size=batch,
            device=device,
        )


def _run_stage4_interactive():
    from imagenet_lc.stages import stage4_evaluate

    print("\n== Stage 4: Evaluation mode ==\n")
    predictions_dir = _prompt("Directory containing predictions JSONs")
    output_dir = _prompt("Directory to save evaluation report", default="eval_output")
    corruptions = _prompt_multi(
        "Which corruptions to include in the CE table?",
        SUPPORTED_CORRUPTIONS,
        default_all=True,
    )
    run_lpips = _prompt_yesno("Compute LPIPS (Table 2)?", default=True)
    clean_dir = None
    corrupted_dir = None
    cap = None
    if run_lpips:
        clean_dir = _prompt("Path to clean dataset (for LPIPS pairing)")
        corrupted_dir = _prompt("Path to corrupted tree (for LPIPS pairing)")
        limit = _prompt(
            "Max LPIPS images per class (blank = all)", default=""
        )
        cap = int(limit) if limit.strip() else None

    stage4_evaluate.run(
        predictions_dir=predictions_dir,
        output_dir=output_dir,
        corruptions=corruptions,
        clean_dir=clean_dir,
        corrupted_dir=corrupted_dir,
        run_lpips=run_lpips,
        lpips_max_images_per_class=cap,
    )


def _run_menu():
    while True:
        print("\n==== ImageNet-LC Pipeline ====")
        print("1. Generate bounding boxes (YOLOv11)")
        print("2. Apply corruptions")
        print("3. Run inference")
        print("4. Evaluation mode (CE, mCE, Relative mCE, LPIPS)")
        print("q. Quit")
        choice = _prompt("Select", default="1")
        if choice in ("q", "Q", "quit", "exit"):
            return
        if choice == "1":
            _run_stage1_interactive()
        elif choice == "2":
            _run_stage2_interactive()
        elif choice == "3":
            _run_stage3_interactive()
        elif choice == "4":
            _run_stage4_interactive()
        else:
            print(f"Unknown choice: {choice!r}")


# ---------------------------------------------------------------------------
# Non-interactive CLI entry
# ---------------------------------------------------------------------------

def _parse_stage1(argv):
    p = argparse.ArgumentParser(prog="main.py --stage 1")
    p.add_argument("--dataset", required=True)
    p.add_argument("--labels", required=True, help="Output labels directory.")
    p.add_argument("--yolo-weights", default="yolo11n.pt")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args(argv)


def _parse_stage2(argv):
    p = argparse.ArgumentParser(prog="main.py --stage 2")
    p.add_argument("--dataset", required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--corruptions", nargs="+", required=True,
        help="Pass 'all' or one/more of: " + ", ".join(SUPPORTED_CORRUPTIONS),
    )
    p.add_argument("--severity", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--bbox-mode", choices=["labels", "random"], default="labels")
    p.add_argument("--num-bboxes", type=int, default=3)
    p.add_argument(
        "--fingerprint-texture",
        default="artifacts/corruptions/fingerprint.jpg",
    )
    p.add_argument("--flare-dir", default="artifacts/corruptions/flares")
    p.add_argument(
        "--illumination-mode", choices=["shadow", "highlight"], default="shadow"
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def _parse_stage3(argv):
    p = argparse.ArgumentParser(prog="main.py --stage 3")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument(
        "--model", required=True,
        help="One of: " + ", ".join(list_models()),
    )
    p.add_argument("--mode", choices=["clean", "corrupted"], default="corrupted")
    p.add_argument("--wnid-source-dir", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def _parse_stage4(argv):
    p = argparse.ArgumentParser(prog="main.py --stage 4")
    p.add_argument("--predictions-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--corruptions", nargs="+", default=list(SUPPORTED_CORRUPTIONS),
    )
    p.add_argument("--clean-dir", default=None)
    p.add_argument("--corrupted-dir", default=None)
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--lpips-max-per-class", type=int, default=None)
    p.add_argument("--lpips-backbone", default="alex")
    p.add_argument("--baseline-model", default="alexnet")
    return p.parse_args(argv)


def _resolve_corruptions(tokens):
    if len(tokens) == 1 and tokens[0].lower() == "all":
        return list(SUPPORTED_CORRUPTIONS)
    unknown = [c for c in tokens if c not in SUPPORTED_CORRUPTIONS]
    if unknown:
        raise SystemExit(
            f"Unknown corruption(s): {unknown}. "
            f"Supported: {list(SUPPORTED_CORRUPTIONS)}"
        )
    return list(tokens)


def _run_stage1_cli(argv):
    from imagenet_lc.stages import stage1_bboxes
    args = _parse_stage1(argv)
    stage1_bboxes.run(
        dataset_dir=args.dataset,
        labels_dir=args.labels,
        yolo_weights=args.yolo_weights,
        top_k=args.top_k,
        skip_existing=args.skip_existing,
    )


def _run_stage2_cli(argv):
    from imagenet_lc.stages import stage2_corrupt
    args = _parse_stage2(argv)
    corruptions = _resolve_corruptions(args.corruptions)
    stage2_corrupt.run(
        dataset_dir=args.dataset,
        labels_dir=args.labels,
        output_dir=args.output,
        corruptions=corruptions,
        severities=args.severity,
        bbox_mode=args.bbox_mode,
        num_random_bboxes=args.num_bboxes,
        fingerprint_texture_path=args.fingerprint_texture,
        flare_dir=args.flare_dir,
        illumination_mode=args.illumination_mode,
        workers=args.workers,
        seed=args.seed,
    )


def _run_stage3_cli(argv):
    from imagenet_lc.stages import stage3_inference
    args = _parse_stage3(argv)
    stage3_inference.run(
        data_dir=args.data_dir,
        output_file=args.output_file,
        model_name=args.model,
        mode=args.mode,
        wnid_source_dir=args.wnid_source_dir,
        batch_size=args.batch_size,
        device=args.device,
    )


def _run_stage4_cli(argv):
    from imagenet_lc.stages import stage4_evaluate
    args = _parse_stage4(argv)
    corruptions = _resolve_corruptions(args.corruptions)
    stage4_evaluate.run(
        predictions_dir=args.predictions_dir,
        output_dir=args.output_dir,
        corruptions=corruptions,
        clean_dir=args.clean_dir,
        corrupted_dir=args.corrupted_dir,
        run_lpips=not args.no_lpips,
        lpips_max_images_per_class=args.lpips_max_per_class,
        lpips_backbone=args.lpips_backbone,
        baseline_model=args.baseline_model,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Tiny front-end parser: just peels off --stage, passes the rest along.
    if len(sys.argv) <= 1:
        _run_menu()
        return

    # Detect --stage flag.
    try:
        idx = sys.argv.index("--stage")
        stage = int(sys.argv[idx + 1])
        argv_rest = sys.argv[1:idx] + sys.argv[idx + 2:]
    except (ValueError, IndexError):
        # No --stage: still offer the menu, but also support bare --help.
        if "--help" in sys.argv or "-h" in sys.argv:
            print(__doc__)
            return
        _run_menu()
        return

    dispatch = {
        1: _run_stage1_cli,
        2: _run_stage2_cli,
        3: _run_stage3_cli,
        4: _run_stage4_cli,
    }
    if stage not in dispatch:
        raise SystemExit(f"Unknown --stage {stage}. Use 1..4.")
    dispatch[stage](argv_rest)


if __name__ == "__main__":
    main()
