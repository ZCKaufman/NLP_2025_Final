import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

MARKER_TYPES = ["Action", "Actor", "Effect", "Evidence", "Victim"]

def load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_div(a, b):
    return a / b if b else 0.0

def get_display_names(paths, names):
    if names and len(names) == len(paths):
        return names
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]

def type_confusion_matrix(report: dict) -> np.ndarray:
    """
    rows: predicted type, cols: gold type
    Note: uses examples['type_confusion'] which may be capped by max_examples_per_bucket.
    """
    conf = np.zeros((len(MARKER_TYPES), len(MARKER_TYPES)), dtype=int)
    items = (report.get("examples") or {}).get("type_confusion", []) or []
    for item in items:
        p = (item.get("pred_span") or {}).get("type")
        g = (item.get("best_overlapping_gold") or {}).get("type")
        if p in MARKER_TYPES and g in MARKER_TYPES:
            conf[MARKER_TYPES.index(p), MARKER_TYPES.index(g)] += 1
    return conf

def boundary_deltas(report: dict):
    """
    Returns (start_deltas, end_deltas) lists from examples['boundary_overlap'].
    Note: also may be capped.
    """
    start_deltas, end_deltas = [], []
    items = (report.get("examples") or {}).get("boundary_overlap", []) or []
    for item in items:
        p = item.get("pred_span") or {}
        g = item.get("best_gold_same_type") or {}
        if all(k in p for k in ("startIndex", "endIndex")) and all(k in g for k in ("startIndex", "endIndex")):
            start_deltas.append(int(p["startIndex"]) - int(g["startIndex"]))
            end_deltas.append(int(p["endIndex"]) - int(g["endIndex"]))
    return start_deltas, end_deltas

def make_bucket_breakdown_fig(reports, names, out_path):
    bucket_order = ["exact_match", "boundary_overlap", "type_confusion", "spurious"]
    bucket_labels = {
        "exact_match": "Exact",
        "boundary_overlap": "Boundary overlap",
        "type_confusion": "Type confusion",
        "spurious": "Spurious",
    }

    n = len(reports)
    fig_w = max(7.5, 4.2 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.8), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, rep, title in zip(axes, reports, names):
        summary = rep["summary"]
        pred_counts = summary["pred_marker_counts"]
        buckets = summary["bucket_counts"]

        mat = np.zeros((len(bucket_order), len(MARKER_TYPES)), dtype=float)
        for j, t in enumerate(MARKER_TYPES):
            denom = pred_counts.get(t, 0)
            for i, b in enumerate(bucket_order):
                mat[i, j] = safe_div(buckets[b].get(t, 0), denom)

        x = np.arange(len(MARKER_TYPES))
        bottom = np.zeros(len(MARKER_TYPES))
        for i, b in enumerate(bucket_order):
            ax.bar(x, mat[i], bottom=bottom, label=bucket_labels[b])
            bottom += mat[i]

        ax.set_xticks(x)
        ax.set_xticklabels(MARKER_TYPES)
        ax.set_ylim(0, 1.0)
        #ax.set_title(title)
        ax.set_ylabel("Proportion of predicted spans")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))

    #fig.suptitle("Predicted-span error breakdown by marker type", y=1.12, fontsize=14)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Write] {out_path}")

def make_recall_proxy_fig(reports, names, out_path):
    n = len(reports)
    fig_w = max(7.0, 3.8 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, rep, title in zip(axes, reports, names):
        summary = rep["summary"]
        gold_counts = summary["gold_marker_counts"]
        buckets = summary["bucket_counts"]

        recall = []
        for t in MARKER_TYPES:
            g = gold_counts.get(t, 0)
            m = buckets["missed"].get(t, 0)
            recall.append(safe_div(g - m, g))

        ax.bar(np.arange(len(MARKER_TYPES)), recall)
        ax.set_xticks(np.arange(len(MARKER_TYPES)))
        ax.set_xticklabels(MARKER_TYPES)
        ax.set_ylim(0, 1.0)
        #ax.set_title(title)
        ax.set_ylabel("Recall proxy (1 - missed/gold)")

    #fig.suptitle("Recall proxy by marker type", fontsize=14)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Write] {out_path}")

def make_type_confusion_fig(reports, names, out_path, annotate=False):
    mats = [type_confusion_matrix(r) for r in reports]
    vmax = max(int(m.max()) for m in mats) if mats else 0

    n = len(reports)
    fig_w = max(7.2, 3.5 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 5.2), constrained_layout=True)
    if n == 1:
        axes = [axes]

    last_im = None
    for ax, mat, title in zip(axes, mats, names):
        last_im = ax.imshow(mat, vmin=0, vmax=vmax if vmax > 0 else None, aspect="equal")
        #ax.set_title(title)
        ax.set_xticks(np.arange(len(MARKER_TYPES)))
        ax.set_yticks(np.arange(len(MARKER_TYPES)))
        ax.set_xticklabels(MARKER_TYPES, rotation=45, ha="right")
        ax.set_yticklabels(MARKER_TYPES)
        ax.set_xlabel("Gold type")
        ax.set_ylabel("Predicted type")

        if annotate:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = int(mat[i, j])
                    if val == 0:
                        continue
                    color = "white" if (vmax > 0 and val / vmax > 0.5) else "black"
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.046, pad=0.02)

    #fig.suptitle("Type confusion: predicted type vs overlapping gold type", fontsize=14)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Write] {out_path}")
    print("[Note] Heatmap uses examples['type_confusion'] (may be capped by max_examples_per_bucket).")

def make_boundary_deltas_fig(reports, names, out_path, bins=30):
    deltas = [boundary_deltas(r) for r in reports]

    all_vals = []
    for sd, ed in deltas:
        all_vals.extend(sd)
        all_vals.extend(ed)

    n = len(reports)
    fig_w = max(7.5, 4.2 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    if len(all_vals) == 0:
        #fig.suptitle("Boundary errors (no boundary_overlap examples found)", fontsize=14)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Write] {out_path}")
        return

    lo = np.percentile(all_vals, 1)
    hi = np.percentile(all_vals, 99)
    pad = max(2, 0.05 * (hi - lo))
    xlim = (lo - pad, hi + pad)

    for ax, (sd, ed), title in zip(axes, deltas, names):
        if len(sd) == 0 and len(ed) == 0:
            ax.text(0.5, 0.5, "No boundary_overlap\nexamples", ha="center", va="center")
            #ax.set_title(title)
            ax.set_axis_off()
            continue

        ax.hist(sd, bins=bins, alpha=0.7, label="Start delta (pred - gold)")
        ax.hist(ed, bins=bins, alpha=0.7, label="End delta (pred - gold)")
        ax.axvline(0, linewidth=1)
        ax.set_xlim(*xlim)
        #ax.set_title(title)
        ax.set_xlabel("Character offset difference")
        ax.set_ylabel("Count")

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
            break

   # fig.suptitle("Boundary errors (char offsets) on overlapping correct-type spans", y=1.12, fontsize=14)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Write] {out_path}")
    print("[Note] Boundary deltas use examples['boundary_overlap'] (may be capped by max_examples_per_bucket).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True, help="Paths to span_error_analysis.json (one per model)")
    ap.add_argument("--names", nargs="+", default=None, help="Optional names (same count as reports)")
    ap.add_argument("--out_dir", default=".", help="Where to write comparison PNGs")
    ap.add_argument("--annotate_heatmap", action="store_true", help="Annotate heatmap cells with counts")
    args = ap.parse_args()

    names = get_display_names(args.reports, args.names)
    reports = [load_report(p) for p in args.reports]

    os.makedirs(args.out_dir, exist_ok=True)

    make_bucket_breakdown_fig(
        reports, names, os.path.join(args.out_dir, "cmp_bucket_breakdown_pred.png")
    )
    make_recall_proxy_fig(
        reports, names, os.path.join(args.out_dir, "cmp_recall_proxy.png")
    )
    make_type_confusion_fig(
        reports, names, os.path.join(args.out_dir, "cmp_type_confusion_heatmap.png"),
        annotate=args.annotate_heatmap
    )
    make_boundary_deltas_fig(
        reports, names, os.path.join(args.out_dir, "cmp_boundary_deltas.png")
    )

if __name__ == "__main__":
    main()
