"""
Corruption Error metrics (Section 4 of the paper).

Implements:

- Top-1 error rate per (model, corruption, severity).
- Corruption Error (CE) per corruption, normalised to AlexNet.   [Eq. 1]
- Mean CE (mCE) across all corruption types.
- Relative CE (accounts for clean accuracy).                     [Eq. 2]
- Relative mCE.

Inputs are predictions JSON files produced by ``stage3_inference.run``.
"""

import json
import os
from collections import defaultdict


BASELINE_MODEL = "alexnet"


def _load_predictions(json_path):
    with open(json_path) as f:
        return json.load(f)


def _error_rate(predictions_dict, key_filter=None):
    """
    Compute top-1 error (fraction of wrong predictions) over a subset of
    ``predictions``. ``key_filter`` is a callable ``str -> bool``.
    """
    total = 0
    wrong = 0
    for key, entry in predictions_dict.items():
        if entry["correct"] is None:
            continue
        if key_filter is not None and not key_filter(key):
            continue
        total += 1
        if not entry["correct"]:
            wrong += 1
    return (wrong / total) if total else None


def per_severity_errors(pred_json, corruption):
    """
    Dict ``{severity: error_rate}`` for a given corruption, from a single
    model's corrupted-mode predictions JSON. Only severities actually
    present in the data are included.
    """
    preds = pred_json["predictions"]
    by_sev = {}
    for sev in ("1", "2", "3", "4", "5"):
        prefix = f"{corruption}/{sev}/"
        e = _error_rate(preds, lambda k: k.startswith(prefix))
        if e is not None:
            by_sev[int(sev)] = e
    return by_sev


def clean_error(pred_json):
    """Top-1 error over the clean split in a clean-mode predictions JSON."""
    return _error_rate(pred_json["predictions"], lambda k: k.startswith("clean/"))


def compute_table(
    model_pred_files,
    corruptions,
    clean_pred_files=None,
    baseline_model=BASELINE_MODEL,
):
    """
    Build the per-model CE and mCE table from a set of predictions files.

    Parameters
    ----------
    model_pred_files : dict
        ``{model_name: path_to_corrupted_predictions.json}``.
    corruptions : list of str
        Corruption types to include (order is preserved in the output).
    clean_pred_files : dict, optional
        ``{model_name: path_to_clean_predictions.json}``. Required for
        Relative CE / Relative mCE (Eq. 2).
    baseline_model : str, optional
        Short name of the baseline model. Defaults to ``'alexnet'``
        (per paper).

    Returns
    -------
    dict
        ``{model_name: {'clean_error': float or None,
                        'per_corruption_CE': {corr: float},
                        'per_corruption_relative_CE': {corr: float or None},
                        'mCE': float,
                        'relative_mCE': float or None}}``
    """
    if baseline_model not in model_pred_files:
        raise SystemExit(
            f"Baseline model '{baseline_model}' is missing from the "
            f"corrupted predictions. Run inference on it first: its error "
            f"rates are the CE denominator (Eq. 1)."
        )

    # Load every JSON once.
    corr_data = {m: _load_predictions(p) for m, p in model_pred_files.items()}
    clean_data = {}
    if clean_pred_files:
        clean_data = {m: _load_predictions(p) for m, p in clean_pred_files.items()}

    # Baseline per-corruption error sums sum_s E_{s,c}^AlexNet.
    baseline_sums = {}
    for c in corruptions:
        per_sev = per_severity_errors(corr_data[baseline_model], c)
        if len(per_sev) == 0:
            raise SystemExit(
                f"Baseline model has no predictions for corruption '{c}'."
            )
        baseline_sums[c] = sum(per_sev.values())

    baseline_clean = (
        clean_error(clean_data[baseline_model])
        if baseline_model in clean_data
        else None
    )

    out = {}
    for model, payload in corr_data.items():
        per_CE = {}
        per_rel_CE = {}
        for c in corruptions:
            per_sev = per_severity_errors(payload, c)
            model_sum = sum(per_sev.values())
            denom = baseline_sums[c]
            ce = (model_sum / denom) if denom else None
            per_CE[c] = ce

            # Relative CE (Eq. 2): requires both clean errors.
            if (
                baseline_clean is not None
                and model in clean_data
                and ce is not None
            ):
                ce_clean = clean_error(clean_data[model])
                if ce_clean is not None:
                    num_rel = model_sum - 5 * ce_clean
                    den_rel = baseline_sums[c] - 5 * baseline_clean
                    per_rel_CE[c] = (
                        (num_rel / den_rel) if den_rel else None
                    )
                else:
                    per_rel_CE[c] = None
            else:
                per_rel_CE[c] = None

        ce_values = [v for v in per_CE.values() if v is not None]
        rel_values = [v for v in per_rel_CE.values() if v is not None]

        out[model] = {
            "clean_error": (
                clean_error(clean_data[model])
                if model in clean_data
                else None
            ),
            "per_corruption_CE": per_CE,
            "per_corruption_relative_CE": per_rel_CE,
            "mCE": (sum(ce_values) / len(ce_values)) if ce_values else None,
            "relative_mCE": (
                sum(rel_values) / len(rel_values) if rel_values else None
            ),
        }
    return out


def format_table(results, corruptions, as_percent=True):
    """Return a human-readable text table similar to paper Table 1."""
    lines = []
    header = (
        ["Model", "ClErr", "mCE", "RelmCE"]
        + [c[:9] for c in corruptions]
    )
    lines.append(" | ".join(f"{h:>8}" for h in header))
    lines.append("-" * len(lines[0]))
    factor = 100.0 if as_percent else 1.0
    for model, r in results.items():
        row = [model]
        row.append(
            f"{r['clean_error'] * factor:.2f}"
            if r["clean_error"] is not None else "--"
        )
        row.append(
            f"{r['mCE'] * factor:.2f}"
            if r["mCE"] is not None else "--"
        )
        row.append(
            f"{r['relative_mCE'] * factor:.2f}"
            if r["relative_mCE"] is not None else "--"
        )
        for c in corruptions:
            v = r["per_corruption_CE"].get(c)
            row.append(f"{v * factor:.2f}" if v is not None else "--")
        lines.append(" | ".join(f"{x:>8}" for x in row))
    return "\n".join(lines)
