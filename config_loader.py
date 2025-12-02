import yaml
import os
from pathlib import Path


class Config:
    def __init__(self, config_path="config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key_path, default=None):
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value
    
    @property
    def model_name(self):
        return self.get('model.name')
    
    @property
    def model_type(self):
        return self.get('model.type')
    
    @property
    def train_file(self):
        return self.get('data.train_file')
    
    @property
    def dev_file(self):
        return self.get('data.dev_file')
    
    @property
    def test_file(self):
        return self.get('data.test_file')
    
    @property
    def submission_file(self):
        return self.get('data.submission_file')
    
    @property
    def batch_size(self):
        return self.get('training.batch_size')
    
    @property
    def eval_batch_size(self):
        return self.get('training.eval_batch_size')
    
    @property
    def inference_batch_size(self):
        return self.get('inference.batch_size')
    
    @property
    def learning_rate(self):
        return self.get('training.learning_rate')
    
    @property
    def num_epochs(self):
        return self.get('training.num_epochs')
    
    @property
    def weight_decay(self):
        return self.get('training.weight_decay')
    
    @property
    def save_strategy(self):
        return self.get('training.save_strategy', 'epoch')
    
    @property
    def save_steps(self):
        return self.get('training.save_steps', 500)
    
    @property
    def save_total_limit(self):
        return self.get('training.save_total_limit', None)
    
    @property
    def gradient_accumulation_steps(self):
        return self.get('training.gradient_accumulation_steps', 1)
    
    @property
    def warmup_ratio(self):
        return self.get('training.warmup_ratio', 0.0)
    
    @property
    def label_to_id(self):
        return self.get('tasks.binary_classification.labels.label_to_id')
    
    @property
    def id_to_label(self):
        id_to_label_dict = self.get('tasks.binary_classification.labels.id_to_label')
        return {int(k): v for k, v in id_to_label_dict.items()}
    
    @property
    def num_labels(self):
        return self.get('tasks.binary_classification.num_labels')
    
    def get_binary_output_dir(self):
        base_dir = self.get('tasks.binary_classification.output_dir')
        model_name_safe = self.model_name.replace('/', '-')
        return f"{base_dir}/{model_name_safe}"
    
    def get_span_output_dir(self, marker_type):
        base_dir = self.get('tasks.span_extraction.output_dir_base')
        model_name_safe = self.model_name.replace('/', '-')
        return f"{base_dir}/{model_name_safe}-{marker_type}"
    
    @property
    def marker_types(self):
        return self.get('tasks.span_extraction.marker_types')
    
    @property
    def max_sequence_length(self):
        return self.get('tasks.span_extraction.max_sequence_length')
    
    @property
    def save_strategy(self):
        return self.get('training.save_strategy', 'epoch')
    
    @property
    def save_total_limit(self):
        return self.get('training.save_total_limit', 2)
    
    @property
    def save_steps(self):
        return self.get('training.save_steps', 500)
    
    def save_config_with_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        config_save_path = os.path.join(output_dir, 'training_config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        print(f"Config saved to: {config_save_path}")


def load_config(config_path="config.yaml"):
    return Config(config_path)