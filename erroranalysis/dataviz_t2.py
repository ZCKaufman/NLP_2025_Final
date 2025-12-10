import argparse
import json
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_report(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_display_names(paths, names):
    if names and len(names) == len(paths):
        return names
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def safe_div(a, b):
    return a / b if b else 0.0


def get_confusion(rep: Dict[str, Any]) -> Tuple[int, int, int, int]:
    c = rep.get("summary", {}).get("confusion", {})
    tp = int(c.get("tp", 0))
    tn = int(c.get("tn", 0))
    fp = int(c.get("fp", 0))
    fn = int(c.get("fn", 0))
    return tp, tn, fp, fn


def has_prob_yes(rep: Dict[str, Any]) -> bool:
    ex = rep.get("examples", {})
    fps = ex.get("false_positives_pred_yes_gold_no", []) or []
    fns = ex.get("false_negatives_pred_no_gold_yes", []) or []
    for item in (fps[:3] + fns[:3]):
        if "prob_yes" in item:
            return True
    return False


def collect_prob_rows(rep: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (y_true, p_yes) arrays using examples lists.
    Uses TP/TN lists if present, otherwise falls back to FP/FN only.
    """
    ex = rep.get("examples", {})

    tp = ex.get("true_positives_pred_yes_gold_yes", []) or []
    tn = ex.get("true_negatives_pred_no_gold_no", []) or []
    fp = ex.get("false_positives_pred_yes_gold_no", []) or []
    fn = ex.get("false_negatives_pred_no_gold_yes", []) or []

    rows = []
    if tp or tn:
        rows.extend(tp)
        rows.extend(tn)
    else:
        # fallback: only errors exist, calibration will be incomplete
        rows.extend(fp)
        rows.extend(fn)

    y_true = []
    p_yes = []
    for r in rows:
        if "prob_yes" not in r:
            continue
        gold = r.get("gold")
        if gold == "Yes":
            y_true.append(1)
        elif gold == "No":
            y_true.append(0)
        else:
            continue
        p_yes.append(float(r["prob_yes"]))
    return np.array(y_true, dtype=int), np.array(p_yes, dtype=float)

def fig_metrics_bar(reports, names, out_path):
    acc = [float(r["summary"].get("accuracy", 0.0)) for r in reports]
    f1m = [float(r["summary"].get("f1_macro", 0.0)) for r in reports]
    f1w = [float(r["summary"].get("f1_weighted", 0.0)) for r in reports]

    x = np.arange(len(reports))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(7.5, 1.8 * len(reports)), 4.6), constrained_layout=True)
    ax.bar(x - w, acc, width=w, label="Accuracy")
    ax.bar(x,     f1m, width=w, label="Macro F1")
    ax.bar(x + w, f1w, width=w, label="Weighted F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    #ax.set_title("Task 2: Overall metrics by model")
    ax.legend(frameon=False)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Write] {out_path}")


def fig_confusion_panels(reports, names, out_path):
    n = len(reports)
    fig_w = max(7.0, 3.6 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.6), constrained_layout=True)
    if n == 1:
        axes = [axes]

    last_im = None
    for ax, rep, title in zip(axes, reports, names):
        tp, tn, fp, fn = get_confusion(rep)
        mat = np.array([[tn, fp],
                        [fn, tp]], dtype=int)

        last_im = ax.imshow(mat, aspect="equal")
        #ax.set_title(title)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred No", "Pred Yes"])
        ax.set_yticklabels(["Gold No", "Gold Yes"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=12)

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.046, pad=0.02)

    #fig.suptitle("Task 2: Confusion matrices", fontsize=14)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Write] {out_path}")


def calibration_curve(y_true, p_yes, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p_yes, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    bin_conf, bin_acc, bin_count = [], [], []
    for b in range(n_bins):
        mask = idx == b
        c = int(np.sum(mask))
        if c == 0:
            continue
        bin_count.append(c)
        bin_conf.append(float(np.mean(p_yes[mask])))
        bin_acc.append(float(np.mean(y_true[mask])))
    return np.array(bin_conf), np.array(bin_acc), np.array(bin_count)


def fig_calibration_panels(reports, names, out_path, n_bins=10):
    # Only include models that actually have prob_yes (and ideally TP/TN lists)
    usable = []
    usable_names = []
    for rep, nm in zip(reports, names):
        if has_prob_yes(rep):
            y, p = collect_prob_rows(rep)
            if len(y) > 0:
                usable.append((rep, y, p))
                usable_names.append(nm)

    if not usable:
        print("[Skip] No reports contain prob_yes; run infer_binary.py with --write_prob_yes (and ideally --include_correct).")
        return

    n = len(usable)
    fig_w = max(7.0, 3.8 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.8), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (rep, y, p), title in zip(axes, usable, usable_names):
        conf, acc, cnt = calibration_curve(y, p, n_bins=n_bins)
        ax.plot([0, 1], [0, 1], linewidth=1)
        ax.plot(conf, acc, marker="o", linewidth=1.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        #ax.set_title(title)
        ax.set_xlabel("Mean predicted P(Yes)")
        ax.set_ylabel("Empirical fraction Yes")

    #fig.suptitle("Task 2: Calibration curves (requires prob_yes)", fontsize=14)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Write] {out_path}")


def fig_error_confidence(reports, names, out_path):
    """
    Histograms of prob_yes for FP vs FN (per model).
    Needs prob_yes in examples.
    """
    usable = []
    usable_names = []
    for rep, nm in zip(reports, names):
        if not has_prob_yes(rep):
            continue
        ex = rep.get("examples", {})
        fp = ex.get("false_positives_pred_yes_gold_no", []) or []
        fn = ex.get("false_negatives_pred_no_gold_yes", []) or []
        fp_p = [float(r["prob_yes"]) for r in fp if "prob_yes" in r]
        fn_p = [float(r["prob_yes"]) for r in fn if "prob_yes" in r]
        if fp_p or fn_p:
            usable.append((fp_p, fn_p))
            usable_names.append(nm)

    if not usable:
        print("[Skip] No error examples contain prob_yes; run infer_binary.py with --write_prob_yes.")
        return

    n = len(usable)
    fig_w = max(7.0, 4.0 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (fp_p, fn_p), title in zip(axes, usable, usable_names):
        ax.hist(fp_p, bins=30, alpha=0.7, label="FP (gold No, pred Yes)")
        ax.hist(fn_p, bins=30, alpha=0.7, label="FN (gold Yes, pred No)")
        #ax.set_title(title)
        ax.set_xlabel("prob_yes")
        ax.set_ylabel("Count")
        ax.legend(frameon=False)

    #fig.suptitle("Task 2: Error confidence distributions", fontsize=14)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Write] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True, help="Paths to task2_error_analysis.json (one per model)")
    ap.add_argument("--names", nargs="+", default=None, help="Optional display names (same count as reports)")
    ap.add_argument("--out_dir", default="task2_viz", help="Where to write PNGs")
    ap.add_argument("--calib_bins", type=int, default=10, help="Calibration bins")
    args = ap.parse_args()

    names = get_display_names(args.reports, args.names)
    reports = [load_report(p) for p in args.reports]
    os.makedirs(args.out_dir, exist_ok=True)

    fig_metrics_bar(reports, names, os.path.join(args.out_dir, "t2_metrics_bar.png"))
    fig_confusion_panels(reports, names, os.path.join(args.out_dir, "t2_confusion_panels.png"))
    fig_error_confidence(reports, names, os.path.join(args.out_dir, "t2_error_confidence.png"))
    fig_calibration_panels(reports, names, os.path.join(args.out_dir, "t2_calibration_panels.png"), n_bins=args.calib_bins)


if __name__ == "__main__":
    main()
