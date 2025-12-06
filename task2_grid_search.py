import yaml
import json
import subprocess
import itertools
import os
import shutil
import time
from datetime import datetime
import csv


class GridSearchRunner:
    def __init__(self, base_config_path="config.yaml", results_dir="experiments"):
        self.base_config_path = base_config_path
        self.results_dir = results_dir
        self.configs_dir = os.path.join(results_dir, "configs")
        self.results_csv = os.path.join(results_dir, "results.csv")
        
        # Create directories
        os.makedirs(self.configs_dir, exist_ok=True)
        
        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    ### IMPORTANT ###
    # Modify the grid here to decide all the hyperparameters that will be searched over
    def define_hyperparameter_grid(self):

        grid = {
            'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
            'batch_size': [16, 32],
            'num_epochs': [10, 15],
            'warmup_ratio': [0.0, 0.1],
            'weight_decay': [0.0, 0.01, 0.1],
            'gradient_accumulation_steps': [1, 2]
        }
        return grid
    
    def generate_experiment_configs(self, grid):
        # Get all combinations
        keys = grid.keys()
        values = grid.values()
        combinations = list(itertools.product(*values))
        
        print(f"Generated {len(combinations)} hyperparameter combinations")
        
        experiments = []
        for idx, combo in enumerate(combinations, start=1):
            exp_id = f"exp{idx:03d}"
            hyperparams = dict(zip(keys, combo))
            
            # Create experiment config
            exp_config = self.base_config.copy()
            
            # Update training hyperparameters
            for param, value in hyperparams.items():
                exp_config['training'][param] = value
            
            # Save experiment config
            config_path = os.path.join(self.configs_dir, f"{exp_id}.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(exp_config, f, default_flow_style=False, sort_keys=False)
            
            experiments.append((exp_id, config_path, hyperparams))
        
        return experiments

    # Does train, infer, and evaluate for all the combinations from the grid    
    def run_experiment(self, exp_id, config_path, hyperparams):
        print(f"\n{'='*70}")
        print(f"Running Experiment: {exp_id}")
        print(f"{'='*70}")
        
        # Get model name from config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_name = config['model']['name']
        
        print(f"Model: {model_name}")
        print(f"Hyperparameters:")
        for param, value in hyperparams.items():
            print(f"  {param}: {value}")
        
        # Copy experiment config to main config.yaml
        shutil.copy(config_path, self.base_config_path)
        
        start_time = time.time()
        
        try:
            # Step 1: Training
            print("\n[1/3] Training model...")
            train_result = subprocess.run(
                ["python", "train_binary.py"],
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if train_result.returncode != 0:
                print(f"ERROR: Training failed for {exp_id}")
                print(train_result.stderr)
                return None
            
            print("✓ Training complete")
            
            # Step 2: Inference
            print("\n[2/3] Running inference...")
            infer_result = subprocess.run(
                ["python", "infer_binary.py"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if infer_result.returncode != 0:
                print(f"ERROR: Inference failed for {exp_id}")
                print(infer_result.stderr)
                return None
            
            print("✓ Inference complete")
            
            # Step 3: Evaluation
            print("\n[3/3] Evaluating...")
            eval_result = subprocess.run(
                ["python", "eval_binary.py",
                 "--reference-file", "val_split.jsonl",
                 "--submission-file", "submission.jsonl",
                 "--output-dir", "./"],
                capture_output=True,
                text=True,
                timeout=240  # 4 minute timeout
            )
            
            if eval_result.returncode != 0:
                print(f"ERROR: Evaluation failed for {exp_id}")
                print(eval_result.stderr)
                return None
            
            print("Evaluation complete")
            
            # Read scores
            with open("scores.json", 'r') as f:
                scores = json.load(f)
            
            training_time = time.time() - start_time
            
            # Prepare results
            results = {
                'exp_id': exp_id,
                'model_name': model_name,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'training_time_seconds': round(training_time, 2),
                'f1_score_weighted': scores.get('f1_score_weighted', 0.0),
                'accuracy': scores.get('accuracy', 0.0),
                'f1_score_yes': scores.get('f1_score_yes', 0.0),
                'f1_score_no': scores.get('f1_score_no', 0.0),
            }
            
            # Add hyperparameters to results
            results.update(hyperparams)
            
            print(f"\n Experiment {exp_id} complete!")
            print(f"  F1 Score (Weighted): {results['f1_score_weighted']:.4f}")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Training Time: {results['training_time_seconds']:.2f}s")
            
            return results
            
        except subprocess.TimeoutExpired:
            print(f"ERROR: Experiment {exp_id} timed out")
            return None
        except Exception as e:
            print(f"ERROR: Experiment {exp_id} failed with exception: {e}")
            return None
    
    def save_results(self, results):
        """Saves or appends results to the CSV file."""
        if not results:
            return
        
        file_exists = os.path.exists(self.results_csv)
        
        with open(self.results_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(results)
        
        print(f"Results saved to {self.results_csv}")
    
    def get_completed_experiments(self):
        if not os.path.exists(self.results_csv):
            return set()
        
        completed = set()
        with open(self.results_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row['exp_id'])
        
        return completed
    
    def run_grid_search(self):
        print("====================")
        print("STARTING GRID SEARCH")
        print("====================")
        
        # Define grid
        grid = self.define_hyperparameter_grid()
        
        print("\nHyperparameter Grid:")
        for param, values in grid.items():
            print(f"  {param}: {values}")
        
        # Generate experiment configs
        experiments = self.generate_experiment_configs(grid)
        
        # Check for completed experiments
        completed = self.get_completed_experiments()
        if completed:
            print(f"\nFound {len(completed)} already completed experiments")
            print(f"Completed: {sorted(completed)}")
            experiments = [(exp_id, cfg, hp) for exp_id, cfg, hp in experiments if exp_id not in completed]
        
        print(f"\nTotal experiments to run: {len(experiments)}")
        
        if len(experiments) == 0:
            print("\n All experiments already completed!")
            self.print_top_results()
            return
        
        # Ask for confirmation
        response = input("\nProceed with grid search? (yes/no): ")
        if response.lower() != 'yes':
            print("Grid search cancelled.")
            return
        
        # Run all experiments
        successful_experiments = 0
        failed_experiments = 0
        
        for exp_id, config_path, hyperparams in experiments:
            results = self.run_experiment(exp_id, config_path, hyperparams)
            
            if results:
                self.save_results(results)
                successful_experiments += 1
            else:
                failed_experiments += 1
        
        # Final summary
        print("\n" + "====================")
        print("GRID SEARCH COMPLETE")
        print("====================")
        print(f"Successful experiments: {successful_experiments}")
        print(f"Failed experiments: {failed_experiments}")
        print(f"\nResults saved to: {self.results_csv}")
        
        if successful_experiments > 0:
            self.print_top_results()
    
    def print_top_results(self, top_n=5):
        if not os.path.exists(self.results_csv):
            return
        
        # Read results
        with open(self.results_csv, 'r') as f:
            reader = csv.DictReader(f)
            results = list(reader)
        
        if not results:
            return
        
        # Sort by F1 score
        results.sort(key=lambda x: float(x['f1_score_weighted']), reverse=True)
        
        print(f"\n{'='*70}")
        print(f"TOP {min(top_n, len(results))} EXPERIMENTS BY F1 SCORE")
        print(f"{'='*70}")
        
        for i, result in enumerate(results[:top_n], 1):
            print(f"\n{i}. {result['exp_id']} - F1: {result['f1_score_weighted']}")
            print(f"   Model: {result['model_name']}")
            print(f"   LR: {result['learning_rate']}, Batch: {result['batch_size']}, "
                  f"Epochs: {result['num_epochs']}, Warmup: {result['warmup_ratio']}")


if __name__ == "__main__":
    print("Grid Search for Hyperparameter Tuning")
    print("=====================================")
    
    runner = GridSearchRunner()
    runner.run_grid_search()