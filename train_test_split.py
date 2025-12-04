import json
import random
from collections import defaultdict


def load_and_filter_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                if 'conspiracy' in item and item['conspiracy'] in ["Yes", "No"]: # Filter out 'Can't tell' labels
                    data.append(item)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    return data


def stratified_split(data, train_ratio=0.9, random_seed=42):
    random.seed(random_seed)
    
    # Group data by label
    label_groups = defaultdict(list)
    for item in data:
        label = item['conspiracy']
        label_groups[label].append(item)
    
    train_data = []
    val_data = []
    
    # Split each label group proportionally
    for label, items in label_groups.items():
        random.shuffle(items)
        
        split_idx = int(len(items) * train_ratio)
        train_data.extend(items[:split_idx])
        val_data.extend(items[split_idx:])
        
        print(f"Label '{label}': {len(items)} total -> {len(items[:split_idx])} train, {len(items[split_idx:])} val")
    
    # Shuffle the combined data
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    return train_data, val_data


def save_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(data)} samples to {file_path}")


if __name__ == "__main__":
    input_file = "train_rehydrated.jsonl"
    train_output = "train_split.jsonl"
    val_output = "val_split.jsonl"
    
    print("Loading and filtering data...")
    data = load_and_filter_data(input_file)
    print(f"Loaded {len(data)} samples after filtering 'Can't tell' labels")
    
    print("\nPerforming stratified split (90% train, 10% val)...")
    train_data, val_data = stratified_split(data, train_ratio=0.9, random_seed=42)
    
    print(f"\nFinal split:")
    print(f"  Training set: {len(train_data)} samples")
    print(f"  Validation set: {len(val_data)} samples")
    
    print("\nSaving splits...")
    save_jsonl(train_data, train_output)
    save_jsonl(val_data, val_output)
    
    print("\n✓ Split complete!")