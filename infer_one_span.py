import json
import sys
import numpy as np
import os
import glob
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    DataCollatorForTokenClassification,
    TrainingArguments,
)
from collections import defaultdict
from config_loader import load_config


def find_latest_checkpoint(base_path, marker_type):
    full_path = f"{base_path}-{marker_type}"
    checkpoint_dirs = glob.glob(os.path.join(full_path, "checkpoint-*"))

    if not checkpoint_dirs:
        print(f"Warning: No 'checkpoint-*' folder found. Assuming model files are in: {full_path}")
        return full_path

    checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split('-')[-1]))

    latest_checkpoint = checkpoint_dirs[-1]
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                item["_id"] = item.get("_id", f"sample_{i}")
                item["text"] = item.get("text", "")
                item["markers"] = item.get("markers", [])
                item["conspiracy"] = item.get("conspiracy", "No")

                data.append(item)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line at index {i} in {file_path}: {line.strip()}")
    print(f"Loaded {len(data)} samples for inference.")
    return data


def tokenize_and_align_labels(examples, tokenizer, label_to_id, max_length):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length,
                                 return_offsets_mapping=True)

    tokenized_inputs["labels"] = [[-100] * len(offset_map) for offset_map in tokenized_inputs["offset_mapping"]]

    return tokenized_inputs


def reconstruct_spans(predictions, tokenized_dataset, id_to_label):
    reconstructed_markers = defaultdict(list)

    positive_label_type = id_to_label.get(1)
    if not positive_label_type or positive_label_type == "O":
        print("Error: Model configuration does not match simplified binary training (ID 1 is not the marker type).")
        return reconstructed_markers

    for i, pred_ids in enumerate(predictions):
        offsets = tokenized_dataset[i]['offset_mapping']
        original_text = tokenized_dataset[i]['text']

        current_span_start_char = None

        for token_idx, label_id in enumerate(pred_ids):
            offset_tuple = offsets[token_idx]

            is_special = (offset_tuple is None or offset_tuple[0] is None or offset_tuple[1] is None or (
                        offset_tuple[0] == 0 and offset_tuple[1] == 0))

            if is_special:
                if current_span_start_char is not None:
                    prev_end_char = None
                    if token_idx > 0 and offsets[token_idx - 1][1] is not None:
                        prev_end_char = offsets[token_idx - 1][1]

                    if prev_end_char is not None:
                        span_text = original_text[current_span_start_char:prev_end_char]
                        reconstructed_markers[i].append({
                            "startIndex": current_span_start_char,
                            "endIndex": prev_end_char,
                            "type": positive_label_type,
                            "text": span_text
                        })

                    current_span_start_char = None
                continue

            label = id_to_label[label_id]
            start_char = offset_tuple[0]

            if label == positive_label_type:
                if current_span_start_char is None:
                    current_span_start_char = start_char

            elif label == 'O':
                if current_span_start_char is not None:
                    prev_end_char = offsets[token_idx - 1][1] if token_idx > 0 and offsets[token_idx - 1][
                        1] is not None else start_char

                    span_text = original_text[current_span_start_char:prev_end_char]
                    reconstructed_markers[i].append({
                        "startIndex": current_span_start_char,
                        "endIndex": prev_end_char,
                        "type": positive_label_type,
                        "text": span_text
                    })
                    current_span_start_char = None

        if current_span_start_char is not None:
            last_valid_end = None
            last_token_idx = len(pred_ids) - 1
            while last_token_idx >= 0:
                offset_tuple_end = offsets[last_token_idx]
                if offset_tuple_end is not None and offset_tuple_end[1] is not None and offset_tuple_end[1] != 0:
                    last_valid_end = offset_tuple_end[1]
                    break
                last_token_idx -= 1

            if last_valid_end is not None:
                span_text = original_text[current_span_start_char:last_valid_end]
                reconstructed_markers[i].append({
                    "startIndex": current_span_start_char,
                    "endIndex": last_valid_end,
                    "type": positive_label_type,
                    "text": span_text
                })

    return reconstructed_markers


if __name__ == '__main__':
    config = load_config()

    raw_data = load_data(config.test_file)
    if not raw_data:
        print("Error: No data loaded. Cannot perform inference.")
        sys.exit(-1)

    unique_ids = [d["_id"] for d in raw_data]
    conspiracy_keys = [d["conspiracy"] for d in raw_data]

    test_dataset = Dataset.from_list(raw_data)

    if config.model_type == "distilbert":
        tokenizer = DistilBertTokenizerFast.from_pretrained(config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    dummy_label_to_id = {"O": 0}

    tokenized_test_dataset = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=[col for col in test_dataset.column_names if
                        col not in ['text', 'offset_mapping', '_id', 'conspiracy']],
        fn_kwargs={"tokenizer": tokenizer, "label_to_id": dummy_label_to_id, "max_length": config.max_sequence_length}
    )

    all_predicted_markers = defaultdict(list)

    for marker_type in config.marker_types:
        model_directory = find_latest_checkpoint(config.get_span_output_dir(marker_type), marker_type)

        print(f"\n{'='*60}")
        print(f"Running inference for marker type: {marker_type}")
        print(f"Model directory: {model_directory}")
        print(f"{'='*60}")

        try:
            if config.model_type == "distilbert":
                model = DistilBertForTokenClassification.from_pretrained(model_directory)
            else:
                model = AutoModelForTokenClassification.from_pretrained(model_directory)

            id_to_label = {0: "O", 1: marker_type}

        except Exception as e:
            print(f"Error loading model for {marker_type} from '{model_directory}'. Details: {e}")
            continue

        data_collator = DataCollatorForTokenClassification(tokenizer)

        prediction_args = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"./tmp_inference_span_{marker_type}",
                per_device_eval_batch_size=config.inference_batch_size,
                report_to="none"
            ),
            data_collator=data_collator,
            tokenizer=tokenizer
        )

        predictions_output = prediction_args.predict(tokenized_test_dataset)

        logits = predictions_output.predictions
        predicted_class_ids = np.argmax(logits, axis=2)

        current_marker_map = reconstruct_spans(predicted_class_ids, tokenized_test_dataset, id_to_label)

        for i, markers in current_marker_map.items():
            all_predicted_markers[i].extend(markers)

    print(f"\n{'='*60}")
    print(f"Saving final aggregated predictions to {config.submission_file}")
    print(f"{'='*60}")

    jsonl_lines = []
    for i in range(len(raw_data)):
        predicted_markers = all_predicted_markers.get(i, [])

        jsonl_obj = {
            "_id": unique_ids[i],
            "conspiracy": conspiracy_keys[i],
            "markers": predicted_markers
        }
        jsonl_lines.append(json.dumps(jsonl_obj))

    with open(config.submission_file, 'w') as f:
        f.write('\n'.join(jsonl_lines) + '\n')

    print(f"Submission file '{config.submission_file}' generated successfully.")