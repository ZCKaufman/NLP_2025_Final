import json
import sys
import numpy as np
import os
import glob
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
    TrainingArguments,
)
from config_loader import load_config

def find_latest_checkpoint(base_path):
    checkpoint_dirs = glob.glob(os.path.join(base_path, "checkpoint-*"))

    if not checkpoint_dirs:
        print(f"Warning: No 'checkpoint-*' folder found. Assuming model files are in: {base_path}")
        return base_path

    checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split('-')[-1]))

    latest_checkpoint = checkpoint_dirs[-1]
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def load_competition_test_data(file_path):
    """
    Loads all data from a JSONL file for inference, preserving order,
    and retaining the document's unique ID.
    """
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                sample_id = item.get("_id", f"sample_{i}")
                data.append({
                    "unique_sample_id": sample_id,
                    "text": item.get("text", "")
                })
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line at index {i} in {file_path}: {line.strip()}")
    print(f"Loaded {len(data)} samples for inference.")
    return data

def tokenize_data(dataset, tokenizer):
    return dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)

if __name__ == '__main__':
    config = load_config()
    
    print(f"Loading configuration...")
    print(f"Model: {config.model_name}")
    print(f"Model Type: {config.model_type}")
    print(f"Test file: {config.test_file}")
    print(f"Model directory: {config.get_binary_output_dir()}")

    raw_data = load_competition_test_data(config.test_file)
    if not raw_data:
        print("Error: No data loaded. Cannot perform inference.")
        sys.exit(-1)

    test_dataset = Dataset.from_list(raw_data)
    unique_ids = test_dataset["unique_sample_id"]

    model_directory = find_latest_checkpoint(config.get_binary_output_dir())

    print(f"Loading model from {model_directory}...")
    try:
        if config.model_type == "distilbert":
            tokenizer = DistilBertTokenizerFast.from_pretrained(config.model_name)
            model = DistilBertForSequenceClassification.from_pretrained(model_directory)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_directory)
    except Exception as e:
        print(f"Error loading model or tokenizer using path: '{model_directory}'.")
        print("Please verify that the directory contains 'config.json' and 'model.safetensors' or 'pytorch_model.bin'.")
        print(f"Details: {e}")
        sys.exit(-1)

    tokenized_test_dataset = tokenize_data(test_dataset, tokenizer)
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(["unique_sample_id", "text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    prediction_args = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_inference",
            per_device_eval_batch_size=config.inference_batch_size,
            report_to="none"
        ),
        data_collator=data_collator
    )

    print("Starting prediction...")
    predictions_output = prediction_args.predict(tokenized_test_dataset)

    logits = predictions_output.predictions
    predicted_class_ids = np.argmax(logits, axis=-1)

    predicted_labels = [config.id_to_label[int(id)] for id in predicted_class_ids]

    print(f"Saving {len(predicted_labels)} predictions to {config.submission_file} (JSONL format)...")

    jsonl_lines = []
    for i, label in enumerate(predicted_labels):
        jsonl_obj = {
            "_id": unique_ids[i],
            "conspiracy": label
        }
        jsonl_lines.append(json.dumps(jsonl_obj))

    with open(config.submission_file, 'w') as f:
        f.write('\n'.join(jsonl_lines) + '\n')

    print(f"Submission file '{config.submission_file}' generated successfully.")
