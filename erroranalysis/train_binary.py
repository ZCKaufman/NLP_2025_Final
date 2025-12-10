import argparse
import json
import inspect
from typing import List, Dict, Any

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

LABEL_TO_ID = {"No": 0, "Yes": 1}
ID_TO_LABEL = {0: "No", 1: "Yes"}


def load_and_filter_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL and keep only rows with conspiracy in {Yes, No} (drop Can't tell)."""
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

            lab = item.get("conspiracy", None)
            if lab in ("Yes", "No"):
                out.append(
                    {
                        "_id": item.get("_id", f"row_{i}"),
                        "text": item.get("text", ""),
                        "conspiracy": lab,
                    }
                )
    print(f"[Data] Loaded {len(out)} labeled (Yes/No) examples from {path}")
    return out


def tokenize_fn(examples, tokenizer, max_length: int):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
    )


def add_labels_fn(examples):
    return {"labels": [LABEL_TO_ID[x] for x in examples["conspiracy"]]}


def compute_metrics(eval_pred):
    """No sklearn dependency. Returns accuracy + macro/weighted F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    labels = labels.astype(int)
    preds = preds.astype(int)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))

    def f1_for_class(pos_label: int):
        if pos_label == 1:
            _tp, _fp, _fn = tp, fp, fn
        else:
            _tp, _fp, _fn = tn, fn, fp

        prec = _tp / max(1, (_tp + _fp))
        rec = _tp / max(1, (_tp + _fn))
        f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
        support = int(np.sum(labels == pos_label))
        return prec, rec, f1, support

    _, _, f10, s0 = f1_for_class(0)
    _, _, f11, s1 = f1_for_class(1)

    macro_f1 = (f10 + f11) / 2.0
    weighted_f1 = (f10 * s0 + f11 * s1) / max(1, (s0 + s1))

    return {
        "accuracy": acc,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "f1_yes": f11,
        "f1_no": f10,
        "support_yes": s1,
        "support_no": s0,
    }


def build_training_args_compat(**raw_kwargs) -> TrainingArguments:
    """
    Transformers version compatibility:
    - Some versions use `evaluation_strategy`, others use `eval_strategy`.
    - This function maps + filters kwargs to what your installed TrainingArguments supports.
    """
    params = inspect.signature(TrainingArguments.__init__).parameters

    # map evaluation arg name
    if "evaluation_strategy" in raw_kwargs and "evaluation_strategy" not in params and "eval_strategy" in params:
        raw_kwargs["eval_strategy"] = raw_kwargs.pop("evaluation_strategy")

    # filter unknown kwargs
    clean_kwargs = {k: v for k, v in raw_kwargs.items() if k in params}
    return TrainingArguments(**clean_kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--val_file", type=str, default=None, help="Optional. If omitted, we do 90/10 split.")
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--output_dir", type=str, default="model-conspiracy-classification")

    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_epochs", type=int, default=10)
    ap.add_argument("--max_length", type=int, default=128)

    ap.add_argument("--warmup_ratio", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    train_rows = load_and_filter_jsonl(args.train_file)
    if len(train_rows) == 0:
        print("Error: no Yes/No examples found in train_file.")
        raise SystemExit(1)

    train_ds = Dataset.from_list(train_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_ds = train_ds.map(add_labels_fn, batched=True)

    if args.val_file:
        val_rows = load_and_filter_jsonl(args.val_file)
        val_ds = Dataset.from_list(val_rows).map(add_labels_fn, batched=True)
        print(f"[Split] Using provided val_file with {len(val_ds)} examples.")
    else:
        split = train_ds.train_test_split(
            test_size=0.10,
            seed=args.seed,
            stratify_by_column="labels",
        )
        train_ds, val_ds = split["train"], split["test"]
        print(f"[Split] 90/10 split -> train={len(train_ds)}, val={len(val_ds)} (stratified)")

    train_tok = train_ds.map(lambda ex: tokenize_fn(ex, tokenizer, args.max_length), batched=True)
    val_tok = val_ds.map(lambda ex: tokenize_fn(ex, tokenizer, args.max_length), batched=True)

    keep_cols = {"input_ids", "attention_mask", "labels"}
    for col in list(train_tok.column_names):
        if col not in keep_cols:
            train_tok = train_tok.remove_columns([col])
    for col in list(val_tok.column_names):
        if col not in keep_cols:
            val_tok = val_tok.remove_columns([col])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = build_training_args_compat(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum_steps,
        fp16=bool(args.fp16),

        # will be mapped to eval_strategy if needed
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,

        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("\n[Train] Training...")
    trainer.train()
    print("[Train] Done.")

    print("\n[Eval] Best checkpoint metrics on val:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        if k.startswith("eval_"):
            print(f"  {k}: {v}")

    print(f"\n[Save] Saving best model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
