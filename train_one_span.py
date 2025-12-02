import json
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset
from config_loader import load_config


def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    return data


def create_label_maps_simplified(marker_type):
    label_list = ["O", marker_type]
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    return label_to_id, id_to_label, len(label_list)


def tokenize_and_align_labels_simplified(examples, tokenizer, label_to_id, marker_type, max_length):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length,
                                 return_offsets_mapping=True)
    labels = []
    all_markers = examples.get("markers", [])

    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        example_labels = [0] * len(offsets)
        example_markers = all_markers[i] if i < len(all_markers) else []

        for marker in example_markers:
            if marker["type"] == marker_type:
                start_char = marker["startIndex"]
                end_char = marker["endIndex"]
                marker_label = label_to_id.get(marker_type)
                if marker_label is not None:
                    for token_idx, (start, end) in enumerate(offsets):
                        if start is not None and end is not None:
                            if start_char <= start < end_char:
                                if token_idx < len(example_labels):
                                    example_labels[token_idx] = marker_label
                            elif start < end_char and end > start_char:
                                if token_idx < len(example_labels) and example_labels[token_idx] == 0:
                                    example_labels[token_idx] = marker_label
        labels.append(example_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


if __name__ == "__main__":
    config = load_config()

    # Load data once
    train_data = load_data(config.train_file)
    train_dataset = Dataset.from_list(train_data)

    if config.model_type == "distilbert":
        tokenizer = DistilBertTokenizerFast.from_pretrained(config.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    for marker_type in config.marker_types:
        print(f"\n{'='*60}")
        print(f"Training model for marker type: {marker_type}")
        print(f"{'='*60}")

        label_to_id, id_to_label, num_labels = create_label_maps_simplified(marker_type)
        print(f"Label mapping: {label_to_id}")

        tokenized_train_dataset = train_dataset.map(
            tokenize_and_align_labels_simplified,
            batched=True,
            fn_kwargs={
                "tokenizer": tokenizer,
                "label_to_id": label_to_id,
                "marker_type": marker_type,
                "max_length": config.max_sequence_length
            }
        )

        if config.model_type == "distilbert":
            model = DistilBertForTokenClassification.from_pretrained(config.model_name, num_labels=num_labels)
        else:
            model = AutoModelForTokenClassification.from_pretrained(config.model_name, num_labels=num_labels)

        output_dir = config.get_span_output_dir(marker_type)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            num_train_epochs=config.num_epochs,
            weight_decay=config.weight_decay,
            save_strategy=config.save_strategy,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_ratio=config.warmup_ratio,
            logging_dir=f'./logs-{marker_type}',
            report_to="none"
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        print(f"\nTraining model for {marker_type}...")
        trainer.train()
        print(f"Training for {marker_type} finished.")
        
        config.save_config_with_model(output_dir)
        print(f"Config saved for {marker_type}")