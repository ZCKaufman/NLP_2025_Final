import os
import json
import argparse
import inspect
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed,
)

MARKER_TYPES_DEFAULT = ["Action", "Actor", "Effect", "Evidence", "Victim"]

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Data] Skipping invalid JSON at line {i}: {line[:120]}...")
                continue

            ex["_id"] = ex.get("_id", f"sample_{i}")
            ex["text"] = ex.get("text", "")
            ex["markers"] = ex.get("markers", []) or []
            data.append(ex)
    return data

def create_label_maps(marker_type):
    label_list = ["O", marker_type]
    label_to_id = {lab: i for i, lab in enumerate(label_list)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}
    return label_to_id, id_to_label


def tokenize_and_align_labels(examples, tokenizer, label_to_id, marker_type, max_length):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
    )

    all_markers = examples.get("markers", [])
    labels = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        ex_labels = [0] * len(offsets)  # default O
        ex_markers = all_markers[i] if i < len(all_markers) else []

        for m in ex_markers:
            if m.get("type") != marker_type:
                continue
            s = m.get("startIndex")
            e = m.get("endIndex")
            if s is None or e is None:
                continue

            pos_id = label_to_id[marker_type]

            for tidx, (ts, te) in enumerate(offsets):
                # skip special/pad
                if ts is None or te is None or ts == te:
                    continue
                # inside span OR overlap
                if (s <= ts < e) or (ts < e and te > s):
                    ex_labels[tidx] = pos_id

        labels.append(ex_labels)

    tokenized.pop("offset_mapping")
    tokenized["labels"] = labels
    return tokenized


def token_f1_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    mask = labels != -100
    labels_m = labels[mask]
    preds_m = preds[mask]

    # binary: positive label is 1
    tp = int(np.sum((preds_m == 1) & (labels_m == 1)))
    fp = int(np.sum((preds_m == 1) & (labels_m == 0)))
    fn = int(np.sum((preds_m == 0) & (labels_m == 1)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = float(np.mean(preds_m == labels_m)) if labels_m.size > 0 else 0.0

    return {
        "precision_token": prec,
        "recall_token": rec,
        "f1_token": f1,
        "accuracy_token": acc,
    }

def reconstruct_spans_one(pred_ids, offsets, text, marker_type):
    spans = []
    start_char = None

    for tidx, lab in enumerate(pred_ids):
        ts, te = offsets[tidx]
        is_special = (ts is None or te is None or (ts == 0 and te == 0))

        if is_special:
            if start_char is not None:
                prev_end = offsets[tidx - 1][1] if tidx > 0 else None
                if prev_end is not None and prev_end >= start_char:
                    spans.append((start_char, prev_end, marker_type))
                start_char = None
            continue

        if lab == 1:
            if start_char is None:
                start_char = ts
        else:
            if start_char is not None:
                prev_end = offsets[tidx - 1][1] if tidx > 0 else ts
                if prev_end is not None and prev_end >= start_char:
                    spans.append((start_char, prev_end, marker_type))
                start_char = None

    if start_char is not None:
        last_end = None
        for j in range(len(offsets) - 1, -1, -1):
            ts, te = offsets[j]
            if te is not None and te != 0:
                last_end = te
                break
        if last_end is not None and last_end >= start_char:
            spans.append((start_char, last_end, marker_type))

    return spans


def exact_span_f1(val_raw, pred_ids_2d, tokenizer, marker_type, max_length):
    gold_set = set()
    pred_set = set()

    for i, ex in enumerate(val_raw):
        gold = ex.get("markers", []) or []
        for g in gold:
            if g.get("type") == marker_type:
                gold_set.add((int(g["startIndex"]), int(g["endIndex"]), marker_type))

        tok = tokenizer(
            ex.get("text", ""),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets = tok["offset_mapping"]
        spans = reconstruct_spans_one(pred_ids_2d[i], offsets, ex.get("text", ""), marker_type)
        for s in spans:
            pred_set.add(s)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    return {"precision_span_exact": prec, "recall_span_exact": rec, "f1_span_exact": f1,
            "tp": tp, "fp": fp, "fn": fn}


def make_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    # map eval_strategy/evaluation_strategy depending on version
    if "eval_strategy" in params and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    return TrainingArguments(**kwargs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="train_split.jsonl")
    ap.add_argument("--val_file", default="val_split.jsonl")
    ap.add_argument("--model_name", default="roberta-base")
    ap.add_argument("--output_dir_base", default="span_models")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--num_epochs", type=int, default=15)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--marker_types", nargs="+", default=MARKER_TYPES_DEFAULT)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir_base, exist_ok=True)

    train_raw = load_jsonl(args.train_file)
    val_raw = load_jsonl(args.val_file)
    if not train_raw:
        raise RuntimeError(f"No training data loaded from {args.train_file}")
    if not val_raw:
        raise RuntimeError(f"No val data loaded from {args.val_file}")

    train_ds = Dataset.from_list(train_raw)
    val_ds = Dataset.from_list(val_raw)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    collator = DataCollatorForTokenClassification(tokenizer)

    for marker_type in args.marker_types:
        print("\n" + "=" * 60)
        print(f"Training marker type: {marker_type}")
        print("=" * 60)

        label_to_id, id_to_label = create_label_maps(marker_type)

        tok_train = train_ds.map(
            tokenize_and_align_labels,
            batched=True,
            fn_kwargs={
                "tokenizer": tokenizer,
                "label_to_id": label_to_id,
                "marker_type": marker_type,
                "max_length": args.max_length,
            },
        )

        tok_val = val_ds.map(
            tokenize_and_align_labels,
            batched=True,
            fn_kwargs={
                "tokenizer": tokenizer,
                "label_to_id": label_to_id,
                "marker_type": marker_type,
                "max_length": args.max_length,
            },
        )

        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            id2label=id_to_label,
            label2id=label_to_id,
        )

        run_dir = os.path.join(
            args.output_dir_base,
            f"{args.model_name.replace('/', '_')}-{marker_type}-L{args.max_length}-lr{args.learning_rate}-bs{args.batch_size}-e{args.num_epochs}-seed{args.seed}",
        )

        training_args = make_training_args(
            output_dir=run_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            report_to="none",
            fp16=bool(args.fp16),
            load_best_model_at_end=True,
            metric_for_best_model="f1_token",
            greater_is_better=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tok_train,
            eval_dataset=tok_val,
            data_collator=collator,
            tokenizer=tokenizer,
            compute_metrics=token_f1_metrics,
        )

        trainer.train()

        eval_metrics = trainer.evaluate()
        print(f"[Eval token metrics] {marker_type}: {eval_metrics}")

        # exact-span F1 on val (character exact match)
        preds_out = trainer.predict(tok_val)
        pred_ids = np.argmax(preds_out.predictions, axis=-1)
        span_metrics = exact_span_f1(val_raw, pred_ids, tokenizer, marker_type, args.max_length)
        print(f"[Eval exact-span metrics] {marker_type}: {span_metrics}")

        # save final
        trainer.save_model(run_dir)
        tokenizer.save_pretrained(run_dir)
        with open(os.path.join(run_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"token_eval": eval_metrics, "span_exact_eval": span_metrics}, f, indent=2)

        print(f"[Saved] {marker_type} model + metrics -> {run_dir}")


if __name__ == "__main__":
    main()
