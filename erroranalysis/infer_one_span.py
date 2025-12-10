import json
import sys
from typing import Dict, List, Any

import numpy as np
import os
import glob
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    DataCollatorForTokenClassification,
    TrainingArguments,
)
from collections import defaultdict

MODEL_PREFIX = "distilbert-base-uncased"

MARKER_TYPES_TO_INFER = ["Action", "Actor", "Effect", "Evidence", "Victim"]
TEST_FILE = "val_split.jsonl"
SUBMISSION_FILE = "submission.jsonl"

MAX_LENGTH = 128
BATCH_SIZE = 64

# error analysis outputs
ERROR_JSON = "electra.json"
ERROR_TXT = "electra.txt"

def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _find_latest_checkpoint_in_dir(run_dir: str) -> str:
    """
    If run_dir has checkpoint-* subfolders, return the latest checkpoint path.
    Otherwise return run_dir.
    """
    ckpts = glob.glob(os.path.join(run_dir, "checkpoint-*"))
    if not ckpts:
        return run_dir

    def ckpt_step(p: str) -> int:
        # checkpoint-1234 -> 1234
        try:
            return int(os.path.basename(p).split("-")[-1])
        except Exception:
            return -1

    ckpts.sort(key=ckpt_step)
    return ckpts[-1]


def find_model_dir_for_type(marker_type: str) -> str:
    """
    Finds a directory matching:
        {MODEL_PREFIX}-{marker_type}*
    and returns either its latest checkpoint or the run dir itself.
    Picks the most recently modified run dir if multiple match.
    """
    base = _script_dir()
    pattern = os.path.join(base, f"{MODEL_PREFIX}-{marker_type}*")
    candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]

    if not candidates:
        raise FileNotFoundError(
            f"No model directory found for type={marker_type} using pattern: {pattern}\n"
            f"Make sure your folder names start with '{MODEL_PREFIX}-{marker_type}'."
        )

    candidates.sort(key=lambda p: os.path.getmtime(p))
    chosen_run_dir = candidates[-1]

    chosen = _find_latest_checkpoint_in_dir(chosen_run_dir)
    return chosen


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads all data from a JSONL file, preserving order, retaining the unique ID."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                item["_id"] = item.get("_id", f"sample_{i}")
                item["text"] = item.get("text", "")
                item["markers"] = item.get("markers", [])
                item["conspiracy"] = item.get("conspiracy", "No")
                data.append(item)
            except json.JSONDecodeError:
                print(f"[Data] Skipping invalid JSON line at index {i}: {line[:120]}...")
    print(f"[Data] Loaded {len(data)} samples from {file_path}")
    return data


def tokenize_for_inference(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )
    # dummy labels so Trainer is happy (ignored in prediction)
    tokenized["labels"] = [[-100] * len(x) for x in tokenized["offset_mapping"]]
    return tokenized

def reconstruct_spans(predictions, tokenized_dataset, id_to_label):
    reconstructed_markers = defaultdict(list)

    positive_label_type = id_to_label.get(1)
    if not positive_label_type or positive_label_type == "O":
        print("[Reconstruct] ERROR: id_to_label[1] is not the marker type.")
        return reconstructed_markers

    for i, pred_ids in enumerate(predictions):
        offsets = tokenized_dataset[i]["offset_mapping"]
        original_text = tokenized_dataset[i]["text"]

        current_span_start_char = None

        for token_idx, label_id in enumerate(pred_ids):
            offset_tuple = offsets[token_idx]

            is_special = (
                offset_tuple is None
                or offset_tuple[0] is None
                or offset_tuple[1] is None
                or (offset_tuple[0] == 0 and offset_tuple[1] == 0)
            )

            if is_special:
                if current_span_start_char is not None:
                    prev_end_char = None
                    if token_idx > 0 and offsets[token_idx - 1][1] is not None:
                        prev_end_char = offsets[token_idx - 1][1]
                    if prev_end_char is not None:
                        span_text = original_text[current_span_start_char:prev_end_char]
                        reconstructed_markers[i].append(
                            {
                                "startIndex": current_span_start_char,
                                "endIndex": prev_end_char,
                                "type": positive_label_type,
                                "text": span_text,
                            }
                        )
                    current_span_start_char = None
                continue

            label = id_to_label[int(label_id)]
            start_char = offset_tuple[0]

            if label == positive_label_type:
                if current_span_start_char is None:
                    current_span_start_char = start_char

            elif label == "O":
                if current_span_start_char is not None:
                    prev_end_char = (
                        offsets[token_idx - 1][1]
                        if token_idx > 0 and offsets[token_idx - 1][1] is not None
                        else start_char
                    )
                    span_text = original_text[current_span_start_char:prev_end_char]
                    reconstructed_markers[i].append(
                        {
                            "startIndex": current_span_start_char,
                            "endIndex": prev_end_char,
                            "type": positive_label_type,
                            "text": span_text,
                        }
                    )
                    current_span_start_char = None

        # finalize if still open
        if current_span_start_char is not None:
            last_valid_end = None
            last_token_idx = len(pred_ids) - 1
            while last_token_idx >= 0:
                off = offsets[last_token_idx]
                if off is not None and off[1] is not None and off[1] != 0:
                    last_valid_end = off[1]
                    break
                last_token_idx -= 1
            if last_valid_end is not None:
                span_text = original_text[current_span_start_char:last_valid_end]
                reconstructed_markers[i].append(
                    {
                        "startIndex": current_span_start_char,
                        "endIndex": last_valid_end,
                        "type": positive_label_type,
                        "text": span_text,
                    }
                )

    return reconstructed_markers


def _overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _is_exact(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (a["startIndex"] == b["startIndex"]) and (a["endIndex"] == b["endIndex"]) and (a["type"] == b["type"])


def _bucket_add(store, key, item, cap):
    if cap is None or len(store[key]) < cap:
        store[key].append(item)




def analyze_span_errors(
    raw_data: List[Dict[str, Any]],
    all_predicted_markers: Dict[int, List[Dict[str, Any]]],
    out_json_path: str = ERROR_JSON,
    out_txt_path: str = ERROR_TXT,
    max_examples_per_bucket: int = None,
) -> Dict[str, Any]:
    marker_types = ["Action", "Actor", "Effect", "Evidence", "Victim"]

    summary = {
        "n_examples": len(raw_data),
        "gold_marker_counts": {t: 0 for t in marker_types},
        "pred_marker_counts": {t: 0 for t in marker_types},
        "bucket_counts": {
            "exact_match": {t: 0 for t in marker_types},
            "boundary_overlap": {t: 0 for t in marker_types},
            "type_confusion": {t: 0 for t in marker_types},
            "spurious": {t: 0 for t in marker_types},
            "missed": {t: 0 for t in marker_types},
        },
        "empty_gold_examples": 0,
        "empty_gold_but_predicted_examples": 0,
    }

    buckets: Dict[str, List[Dict[str, Any]]] = {
        "exact_match": [],
        "boundary_overlap": [],
        "type_confusion": [],
        "spurious": [],
        "missed": [],
        "empty_gold_but_predicted": [],
    }

    for i, ex in enumerate(raw_data):
        ex_id = ex.get("_id", f"idx_{i}")
        text = ex.get("text", "")
        gold = ex.get("markers", []) or []
        pred = all_predicted_markers.get(i, []) or []

        # --- Counts ---
        for g in gold:
            if g.get("type") in marker_types:
                summary["gold_marker_counts"][g["type"]] += 1
        for p in pred:
            if p.get("type") in marker_types:
                summary["pred_marker_counts"][p["type"]] += 1

        # --- Empty-gold bookkeeping ---
        if len(gold) == 0:
            summary["empty_gold_examples"] += 1
            if len(pred) > 0:
                summary["empty_gold_but_predicted_examples"] += 1
                _bucket_add(
                    buckets,
                    "empty_gold_but_predicted",
                    {
                        "_id": ex_id,
                        "text": text,         # full text
                        "gold_markers": [],   # none
                        "pred_markers": pred, # all predicted markers
                        "note": "No gold markers, but model predicted at least one marker.",
                    },
                    max_examples_per_bucket,
                )

        gold_by_type = {t: [] for t in marker_types}
        for g in gold:
            t = g.get("type")
            if t in gold_by_type:
                gold_by_type[t].append(g)

        gold_all = [g for g in gold if g.get("type") in marker_types]

        # --- Classify each predicted span ---
        for p in pred:
            p_type = p.get("type")
            if p_type not in marker_types:
                continue

            overlaps_any = []
            for g in gold_all:
                ov = _overlap_len(p["startIndex"], p["endIndex"], g["startIndex"], g["endIndex"])
                if ov > 0:
                    overlaps_any.append((ov, g))

            # 1) Spurious: no overlap with any gold
            if not overlaps_any:
                summary["bucket_counts"]["spurious"][p_type] += 1
                _bucket_add(
                    buckets,
                    "spurious",
                    {
                        "_id": ex_id,
                        "text": text,
                        "pred_span": p,
                        "gold_markers": gold,  # all gold for context
                        "note": f"Predicted {p_type} span does not overlap any gold span.",
                    },
                    max_examples_per_bucket,
                )
                continue

            # Overlaps at least one gold span
            overlaps_same_type = []
            for g in gold_by_type[p_type]:
                ov = _overlap_len(p["startIndex"], p["endIndex"], g["startIndex"], g["endIndex"])
                if ov > 0:
                    overlaps_same_type.append((ov, g))

            is_exact_any = any(_is_exact(p, g) for _, g in overlaps_same_type)

            if overlaps_same_type:
                # 2) Exact match
                if is_exact_any:
                    summary["bucket_counts"]["exact_match"][p_type] += 1
                    _bucket_add(
                        buckets,
                        "exact_match",
                        {
                            "_id": ex_id,
                            "text": text,
                            "pred_span": p,
                            "gold_overlaps_same_type": [g for _, g in overlaps_same_type],
                            "note": f"Exact match for {p_type}.",
                        },
                        max_examples_per_bucket,
                    )
                # 3) Boundary overlap with same type
                else:
                    summary["bucket_counts"]["boundary_overlap"][p_type] += 1
                    best_g = max(overlaps_same_type, key=lambda x: x[0])[1]
                    _bucket_add(
                        buckets,
                        "boundary_overlap",
                        {
                            "_id": ex_id,
                            "text": text,
                            "pred_span": p,
                            "best_gold_same_type": best_g,
                            "note": f"Overlaps correct {p_type} gold span but boundaries differ.",
                        },
                        max_examples_per_bucket,
                    )
            else:
                # 4) Overlaps only gold spans of *other* types -> type confusion
                best_ov, best_g = max(overlaps_any, key=lambda x: x[0])
                gold_type = best_g.get("type")
                summary["bucket_counts"]["type_confusion"][p_type] += 1
                _bucket_add(
                    buckets,
                    "type_confusion",
                    {
                        "_id": ex_id,
                        "text": text,
                        "pred_span": p,
                        "best_overlapping_gold": best_g,
                        "note": f"Predicted {p_type} overlaps gold {gold_type} (overlap={best_ov}).",
                    },
                    max_examples_per_bucket,
                )

        pred_by_type = {t: [] for t in marker_types}
        for p in pred:
            t = p.get("type")
            if t in pred_by_type:
                pred_by_type[t].append(p)

        for t in marker_types:
            for g in gold_by_type[t]:
                has_overlap_same = False
                for p in pred_by_type[t]:
                    if _overlap_len(p["startIndex"], p["endIndex"], g["startIndex"], g["endIndex"]) > 0:
                        has_overlap_same = True
                        break
                if not has_overlap_same:
                    summary["bucket_counts"]["missed"][t] += 1
                    _bucket_add(
                        buckets,
                        "missed",
                        {
                            "_id": ex_id,
                            "text": text,
                            "missed_gold_span": g,
                            "pred_markers_same_type": pred_by_type[t],
                            "note": f"Gold {t} span has no overlapping predicted {t} span.",
                        },
                        max_examples_per_bucket,
                    )

    report = {"summary": summary, "examples": buckets}

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("=== SPAN ERROR ANALYSIS SAMPLES (FULL DATA) ===\n\n")
        for bucket_name, items in buckets.items():
            f.write(f"\n\n## {bucket_name.upper()} (showing {len(items)} examples)\n")
            for it in items:
                f.write(f"\n_id: {it.get('_id')}\n")
                f.write(f"text: {it.get('text','')}\n")
                for k in ["pred_span", "missed_gold_span", "best_gold_same_type", "best_overlapping_gold"]:
                    if k in it:
                        f.write(f"{k}: {json.dumps(it[k], ensure_ascii=False)}\n")
                if "gold_markers" in it:
                    f.write(f"gold_markers: {json.dumps(it['gold_markers'], ensure_ascii=False)}\n")
                if "pred_markers" in it:
                    f.write(f"pred_markers: {json.dumps(it['pred_markers'], ensure_ascii=False)}\n")
                if "pred_markers_same_type" in it:
                    f.write(f"pred_markers_same_type: {json.dumps(it['pred_markers_same_type'], ensure_ascii=False)}\n")
                f.write(f"note: {it.get('note','')}\n")
                f.write("-" * 60 + "\n")

    print(f"[ErrorAnalysis] Wrote: {out_json_path}")
    print(f"[ErrorAnalysis] Wrote: {out_txt_path}")

    return report



if __name__ == "__main__":
    raw_data = load_data(os.path.join(_script_dir(), TEST_FILE))
    if not raw_data:
        print("Error: No data loaded. Cannot perform inference.")
        sys.exit(-1)

    unique_ids = [d["_id"] for d in raw_data]
    conspiracy_keys = [d["conspiracy"] for d in raw_data]

    test_dataset = Dataset.from_list(raw_data)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PREFIX, use_fast=True)

    tokenized_test_dataset = test_dataset.map(
        lambda ex: tokenize_for_inference(ex, tokenizer),
        batched=True,
    )
    keep_cols = {"input_ids", "attention_mask", "labels", "offset_mapping", "text"}
    drop_cols = [c for c in tokenized_test_dataset.column_names if c not in keep_cols]
    if drop_cols:
        tokenized_test_dataset = tokenized_test_dataset.remove_columns(drop_cols)

    all_predicted_markers = defaultdict(list)
    loaded_any = False

    for marker_type in MARKER_TYPES_TO_INFER:
        print("\n" + "=" * 60)
        print(f"=== Inference for marker type: {marker_type} ===")

        try:
            model_dir = find_model_dir_for_type(marker_type)
            print(f"[Model] Loading from: {model_dir}")
            model = AutoModelForTokenClassification.from_pretrained(model_dir)
            loaded_any = True
        except Exception as e:
            print(f"[Model] ERROR loading {marker_type} model: {e}")
            continue

        id_to_label = {0: "O", 1: marker_type}  # binary model assumption

        data_collator = DataCollatorForTokenClassification(tokenizer)

        predictor = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=os.path.join(_script_dir(), f"./tmp_inference_span_{marker_type}"),
                per_device_eval_batch_size=BATCH_SIZE,
                report_to="none",
            ),
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        preds = predictor.predict(tokenized_test_dataset)
        logits = preds.predictions
        pred_ids = np.argmax(logits, axis=2)

        current_marker_map = reconstruct_spans(pred_ids, tokenized_test_dataset, id_to_label)

        for i, markers in current_marker_map.items():
            all_predicted_markers[i].extend(markers)

    if not loaded_any:
        print("\n[WARN] No models were loaded successfully, so predictions will be empty.\n")

    print(f"\n[Write] Writing predictions to {SUBMISSION_FILE}")
    jsonl_lines = []
    for i in range(len(raw_data)):
        jsonl_obj = {
            "_id": unique_ids[i],
            "conspiracy": conspiracy_keys[i],
            "markers": all_predicted_markers.get(i, []),
        }
        jsonl_lines.append(json.dumps(jsonl_obj))

    with open(os.path.join(_script_dir(), SUBMISSION_FILE), "w", encoding="utf-8") as f:
        f.write("\n".join(jsonl_lines) + "\n")
    print(f"[Write] Done: {SUBMISSION_FILE}")

    analyze_span_errors(
        raw_data=raw_data,
        all_predicted_markers=all_predicted_markers,
        out_json_path=os.path.join(_script_dir(), ERROR_JSON),
        out_txt_path=os.path.join(_script_dir(), ERROR_TXT),
        max_examples_per_bucket=None,
    )
