import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from config_loader import load_config

def load_and_filter_data(file_path):
    """Loads data from a JSON file and filters out entries with 'Can't tell'."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                if 'conspiracy' in item and item['conspiracy'] in ["Yes", "No"]:
                    data.append(item)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    return data


def tokenize_data(dataset, tokenizer):
    """Tokenizes the text data."""
    return dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)


def encode_labels(dataset, label_to_id):
    """Encodes the labels to numerical values."""
    return dataset.map(lambda examples: {'labels': [label_to_id[label] for label in examples["conspiracy"]]},
                       batched=True)

if __name__ == "__main__":
    config = load_config()
    
    print(f"Loading configuration...")
    print(f"Model: {config.model_name}")
    print(f"Model Type: {config.model_type}")
    print(f"Training file: {config.train_file}")
    print(f"Output directory: {config.get_binary_output_dir()}")

    # Load and filter data
    train_data = load_and_filter_data(config.train_file)
    print(f"Loaded {len(train_data)} training samples")

    train_dataset = Dataset.from_list(train_data)

    # Load tokenizer and model using config
    if config.model_type == "distilbert":
        tokenizer = DistilBertTokenizerFast.from_pretrained(config.model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            id2label=config.id_to_label,
            label2id=config.label_to_id
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            id2label=config.id_to_label,
            label2id=config.label_to_id
        )
    
    tokenized_train_dataset = tokenize_data(train_dataset, tokenizer)
    encoded_train_dataset = encode_labels(tokenized_train_dataset, config.label_to_id)

    output_dir = config.get_binary_output_dir()

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        logging_dir='./logs',
        report_to="none"
    )

    
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train_dataset,
        tokenizer=tokenizer
    )
    
    # Train the model
    print("Training the model...")
    trainer.train()
    print("Training finished.")

    # Save the config so we know what parameters we used
    config.save_config_with_model(output_dir) 