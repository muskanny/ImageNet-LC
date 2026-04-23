"""
Stage 4: Evaluation mode.

Given a set of predictions JSON files (from Stage 3) and optionally the
clean / corrupted image trees, compute all reported metrics:

    - Top-1 error (clean and per corruption-severity).
    - Corruption Error (CE), Mean CE (mCE), Relative CE, Relative mCE
      (paper Table 1).
    - LPIPS per severity, averaged across corruptions (paper Table 2).

Results are written as JSON and as a human-readable text report.
"""

import json
import os

from ..metrics import corruption_error, lpips_metric


def _collect_prediction_files(
    predictions_dir, expected_suffix_corr="_corrupted.json",
    expected_suffix_clean="_clean.json",
):
    """
    Find predictions JSONs in a directory following the naming convention
    that ``run`` uses by default:

        <predictions_dir>/<model>_corrupted.json
        <predictions_dir>/<model>_clean.json
    """
    corr = {}
    clean = {}
    if not os.path.isdir(predictions_dir):
        return corr, clean
    for fname in os.listdir(predictions_dir):
        full = os.path.join(predictions_dir, fname)
        if fname.endswith(expected_suffix_corr):
            model = fname[: -len(expected_suffix_corr)]
            corr[model] = full
        elif fname.endswith(expected_suffix_clean):
            model = fname[: -len(expected_suffix_clean)]
            clean[model] = full
    return corr, clean


def run(
    predictions_dir,
    output_dir,
    corruptions,
    clean_dir=None,
    corrupted_dir=None,
    run_lpips=True,
    lpips_max_images_per_class=None,
    lpips_backbone="alex",
    baseline_model=corruption_error.BASELINE_MODEL,
):
    """
    Evaluation mode: compute every metric the paper reports.

    Parameters
    ----------
    predictions_dir : str
        Directory containing per-model predictions JSONs produced by
        Stage 3. Expected filename convention:
        ``<model>_corrupted.json`` and (optionally) ``<model>_clean.json``.
    output_dir : str
        Destination for the evaluation summary (``eval_results.json``)
        and the text report (``eval_report.txt``).
    corruptions : list of str
        Corruption types to include in the CE table (in paper order).
    clean_dir : str, optional
        Path to the clean ImageNet dataset. Required for LPIPS.
    corrupted_dir : str, optional
        Path to the corrupted tree. Required for LPIPS.
    run_lpips : bool, optional
        Whether to compute LPIPS (paper Table 2). Default True.
    lpips_max_images_per_class : int, optional
        Cap LPIPS pair count per class for speed.
    lpips_backbone : str, optional
        LPIPS backbone ('alex' matches the paper).
    baseline_model : str, optional
        Short name of the baseline model used in Eq. 1 / 2.
    """
    os.makedirs(output_dir, exist_ok=True)


    # --- Corruption-error metrics ---------------------------------------
    corr_files, clean_files = _collect_prediction_files(predictions_dir)
    if not corr_files:
        raise SystemExit(
            f"No <model>_corrupted.json files found under {predictions_dir}. "
            "Run Stage 3 first."
        )
    if baseline_model not in corr_files:
        raise SystemExit(
            f"Baseline model '{baseline_model}' is missing from the corrupted predictions. "
            f"Run inference on it first: its error rates are the CE denominator (Eq. 1)."
        )
    if baseline_model not in clean_files:
        raise SystemExit(
            f"Baseline model '{baseline_model}' is missing from the clean predictions. "
            f"Run inference on it first: its clean accuracy is required for relative metrics (Eq. 2)."
        )

    print(
        f"Evaluation mode: found corrupted predictions for "
        f"{len(corr_files)} model(s), clean predictions for "
        f"{len(clean_files)} model(s)."
    )

    if corrupted_dir:
        corruptions = sorted(os.listdir(corrupted_dir))

    ce_results = corruption_error.compute_table(
        model_pred_files=corr_files,
        corruptions=corruptions,
        clean_pred_files=clean_files,
        baseline_model=baseline_model,
    )
    ce_table_text = corruption_error.format_table(ce_results, corruptions)

    # --- LPIPS ----------------------------------------------------------
    lpips_results = None
    if run_lpips:
        if not clean_dir or not corrupted_dir:
            print(
                "[info] skipping LPIPS: both --clean-dir and --corrupted-dir "
                "are required."
            )
        else:
            lpips_results = lpips_metric.compute(
                clean_dir=clean_dir,
                corrupted_dir=corrupted_dir,
                net=lpips_backbone,
                max_images_per_class=lpips_max_images_per_class,
                corruption_filter=corruptions,
            )

    # --- Persist --------------------------------------------------------
    payload = {
        "corruption_error": {
            "baseline_model": baseline_model,
            "corruptions": corruptions,
            "per_model": ce_results,
        },
        "lpips": lpips_results,
    }
    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    report_lines = [
        "=" * 72,
        "ImageNet-LC Evaluation Report",
        "=" * 72,
        "",
        "1. Corruption Error (values are errors normalised to baseline, in %).",
        "",
        ce_table_text,
        "",
    ]
    if lpips_results is not None:
        report_lines.extend(
            [
                "2. LPIPS (severity-wise, averaged across corruptions).",
                "",
                lpips_metric.format_severity_table(
                    lpips_results["per_severity"]
                ),
                "",
                f"   (backbone: {lpips_results['backbone']}, "
                f"{lpips_results['n_pairs']} pairs)",
            ]
        )
    report_text = "\n".join(report_lines)

    report_path = os.path.join(output_dir, "eval_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved text report: {report_path}")
