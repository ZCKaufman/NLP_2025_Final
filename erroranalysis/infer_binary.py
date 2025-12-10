import argparse
import json
import os
import glob
from typing import List, Dict, Any, Optional

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)


def find_latest_checkpoint(base_path: str) -> str:
    ckpts = glob.glob(os.path.join(base_path, "checkpoint-*"))
    if not ckpts:
        print(f"[Checkpoint] No checkpoint-* found; using: {base_path}")
        return base_path
    ckpts.sort(key=lambda x: int(os.path.basename(x).split("-")[-1]))
    latest = ckpts[-1]
    print(f"[Checkpoint] Latest: {latest}")
    return latest


def load_jsonl_for_inference(path: str) -> List[Dict[str, Any]]:
    """Load JSONL and keep _id, text, and (optional) gold conspiracy label."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Data] Skipping invalid JSON line {i}: {line[:200]}")
                continue

            out.append(
                {
                    "_id": item.get("_id", f"sample_{i}"),
                    "text": item.get("text", ""),
                    # gold may or may not exist
                    "conspiracy": item.get("conspiracy", None),
                }
            )
    print(f"[Data] Loaded {len(out)} samples from {path}")
    return out


def tokenize_fn(examples, tokenizer, max_length: int):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
    )


def analyze_binary_errors(
    raw: List[Dict[str, Any]],
    pred_ids: np.ndarray,
    id2label: Dict[int, str],
    out_json: str,
    out_txt: str,
    probs_yes: Optional[np.ndarray] = None,
    include_correct: bool = False,   # NEW: optionally include TP/TN lists too
    sort_by_confidence: bool = True, # NEW: sort error lists by prob_yes if available
) -> None:
    gold = [r.get("conspiracy") for r in raw]
    if not all(g in ("Yes", "No", "Can't tell") for g in gold):
        print("[ErrorAnalysis] No gold Yes/No labels found in test_file. Skipping error analysis.")
        return

    gold_ids = np.array([1 if g == "Yes" else 0 for g in gold], dtype=int)
    pred_ids = pred_ids.astype(int)

    tp = int(np.sum((pred_ids == 1) & (gold_ids == 1)))
    tn = int(np.sum((pred_ids == 0) & (gold_ids == 0)))
    fp = int(np.sum((pred_ids == 1) & (gold_ids == 0)))
    fn = int(np.sum((pred_ids == 0) & (gold_ids == 1)))

    acc = (tp + tn) / max(1, tp + tn + fp + fn)

    prec_yes = tp / max(1, tp + fp)
    rec_yes = tp / max(1, tp + fn)
    f1_yes = 0.0 if (prec_yes + rec_yes) == 0 else 2 * prec_yes * rec_yes / (prec_yes + rec_yes)

    prec_no = tn / max(1, tn + fn)
    rec_no = tn / max(1, tn + fp)
    f1_no = 0.0 if (prec_no + rec_no) == 0 else 2 * prec_no * rec_no / (prec_no + rec_no)

    macro_f1 = (f1_yes + f1_no) / 2.0
    s_yes = int(np.sum(gold_ids == 1))
    s_no = int(np.sum(gold_ids == 0))
    weighted_f1 = (f1_yes * s_yes + f1_no * s_no) / max(1, s_yes + s_no)

    fp_examples: List[Dict[str, Any]] = []
    fn_examples: List[Dict[str, Any]] = []
    tp_examples: List[Dict[str, Any]] = []
    tn_examples: List[Dict[str, Any]] = []

    for i, r in enumerate(raw):
        pred_lab = id2label[int(pred_ids[i])]
        gold_lab = r.get("conspiracy")

        payload = {
            "_id": r["_id"],
            "gold": gold_lab,
            "pred": pred_lab,
            "text": r["text"],  # FULL TEXT (no truncation)
        }
        if probs_yes is not None:
            payload["prob_yes"] = float(probs_yes[i])

        if gold_lab == "No" and pred_lab == "Yes":
            fp_examples.append(payload)
        elif gold_lab == "Yes" and pred_lab == "No":
            fn_examples.append(payload)
        elif include_correct and gold_lab == "Yes" and pred_lab == "Yes":
            tp_examples.append(payload)
        elif include_correct and gold_lab == "No" and pred_lab == "No":
            tn_examples.append(payload)

    # Optional sorting by confidence (if prob_yes is available)
    if sort_by_confidence and probs_yes is not None:
        # FP: predicted Yes with high confidence
        fp_examples.sort(key=lambda x: x.get("prob_yes", 0.0), reverse=True)
        # FN: predicted No; these are often "should be Yes" but model prob_yes is low
        fn_examples.sort(key=lambda x: x.get("prob_yes", 1.0))
        if include_correct:
            # TP: high prob_yes means strongly correct Yes
            tp_examples.sort(key=lambda x: x.get("prob_yes", 0.0), reverse=True)
            # TN: low prob_yes means strongly correct No
            tn_examples.sort(key=lambda x: x.get("prob_yes", 1.0))

    report = {
        "summary": {
            "n": len(raw),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "accuracy": acc,
            "f1_yes": f1_yes,
            "f1_no": f1_no,
            "f1_macro": macro_f1,
            "f1_weighted": weighted_f1,
            "support_yes": s_yes,
            "support_no": s_no,
        },
        "examples": {
            "false_positives_pred_yes_gold_no": fp_examples,
            "false_negatives_pred_no_gold_yes": fn_examples,
        },
    }

    if include_correct:
        report["examples"].update({
            "true_positives_pred_yes_gold_yes": tp_examples,
            "true_negatives_pred_no_gold_no": tn_examples,
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=== TASK 2 ERROR ANALYSIS (FULL DATA) ===\n\n")
        f.write(f"ACC={acc:.4f}  macroF1={macro_f1:.4f}  weightedF1={weighted_f1:.4f}\n")
        f.write(f"TP={tp} TN={tn} FP={fp} FN={fn}\n")
        f.write(f"support_yes={s_yes} support_no={s_no}\n\n")

        def dump_list(title: str, items: List[Dict[str, Any]]):
            f.write(f"## {title} (n={len(items)})\n")
            for ex in items:
                f.write(f"\n_id: {ex['_id']}\n")
                if "prob_yes" in ex:
                    f.write(f"prob_yes: {ex['prob_yes']:.6f}\n")
                f.write(f"text: {ex['text']}\n")
                f.write("-" * 60 + "\n")
            f.write("\n")

        dump_list("FALSE POSITIVES (gold=No, pred=Yes)", fp_examples)
        dump_list("FALSE NEGATIVES (gold=Yes, pred=No)", fn_examples)

        if include_correct:
            dump_list("TRUE POSITIVES (gold=Yes, pred=Yes)", tp_examples)
            dump_list("TRUE NEGATIVES (gold=No, pred=No)", tn_examples)

    print(f"[ErrorAnalysis] Wrote: {out_json}")
    print(f"[ErrorAnalysis] Wrote: {out_txt}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, help="Folder like roberta_task2_run_seed42 (or base output dir)")
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Tokenizer base (usually same family)")
    ap.add_argument("--test_file", type=str, required=True)
    ap.add_argument("--submission_file", type=str, default="submission_task2.jsonl")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--include_correct", action="store_true", help="Also store TP/TN full lists in the report.")
    ap.add_argument("--no_sort_by_confidence", action="store_true", help="Disable confidence-based sorting of examples.")


    ap.add_argument("--error_json", type=str, default="task2_error_analysis.json")
    ap.add_argument("--error_txt", type=str, default="task2_error_samples.txt")
    ap.add_argument("--max_error_examples", type=int, default=50)
    ap.add_argument("--write_error_analysis", action="store_true", help="If set, attempt error analysis when gold exists.")
    ap.add_argument("--write_prob_yes", action="store_true", help="If set, include prob_yes in error samples (requires softmax).")
    args = ap.parse_args()

    model_dir = find_latest_checkpoint(args.model_dir)

    print(f"[Load] tokenizer={args.model_name}  model={model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    raw = load_jsonl_for_inference(args.test_file)
    if not raw:
        raise SystemExit("[Data] No rows loaded; aborting.")

    ds = Dataset.from_list(raw)
    tok = ds.map(lambda ex: tokenize_fn(ex, tokenizer, args.max_length), batched=True)

    keep_cols = {"input_ids", "attention_mask", "token_type_ids"}
    drop_cols = [c for c in tok.column_names if c not in keep_cols]
    if drop_cols:
        tok = tok.remove_columns(drop_cols)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    predictor = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_inference_task2",
            per_device_eval_batch_size=args.batch_size,
            report_to="none",
        ),
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("[Infer] Predicting...")
    preds = predictor.predict(tok)
    logits = preds.predictions

    pred_ids = np.argmax(logits, axis=-1).astype(int)

    id2label = getattr(model.config, "id2label", None)
    if not isinstance(id2label, dict) or set(map(int, id2label.keys())) != {0, 1}:
        id2label = {0: "No", 1: "Yes"}
    else:
        id2label = {int(k): v for k, v in id2label.items()}

    pred_labels = [id2label[int(i)] for i in pred_ids]

    print(f"[Write] Writing {len(pred_labels)} rows to {args.submission_file}")
    with open(args.submission_file, "w", encoding="utf-8") as f:
        for i, lab in enumerate(pred_labels):
            f.write(json.dumps({"_id": raw[i]["_id"], "conspiracy": lab}, ensure_ascii=False) + "\n")
    print("[Write] Done.")

    if args.write_error_analysis:
        probs_yes = None
        if args.write_prob_yes:
            # stable softmax for binary logits
            m = np.max(logits, axis=-1, keepdims=True)
            exp = np.exp(logits - m)
            probs = exp / np.sum(exp, axis=-1, keepdims=True)
            probs_yes = probs[:, 1].astype(float)

        analyze_binary_errors(
            raw=raw,
            pred_ids=pred_ids,
            id2label=id2label,
            out_json=args.error_json,
            out_txt=args.error_txt,
            probs_yes=probs_yes,
            include_correct=bool(args.include_correct),
            sort_by_confidence=not bool(args.no_sort_by_confidence),
        )



if __name__ == "__main__":
    main()
