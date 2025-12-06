import yaml
import json
import subprocess
import itertools
import os
import shutil
import time
from datetime import datetime
import csv


class SpanGridSearchRunner:
    def __init__(self, base_config_path="config.yaml", results_dir="experiments_span"):
        self.base_config_path = base_config_path
        self.results_dir = results_dir
        self.configs_dir = os.path.join(results_dir, "configs")
        self.results_csv = os.path.join(results_dir, "results.csv")
        
        # Create directories
        os.makedirs(self.configs_dir, exist_ok=True)
        
        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def define_hyperparameter_grid(self):
        """
        Define the hyperparameter grid for span extraction.
        Note: Training takes ~5x longer than binary (5 models vs 1)
        """
        # Optimized for Fedora desktop with 96GB RAM and 2x RTX 4070 Ti Super
        # Smaller grid than binary due to 5x longer training time
        grid = {
            'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
            'batch_size': [16, 32],
            'num_epochs': [10, 15],
        }
        return grid
    
    def generate_experiment_configs(self, grid):
        """Generate all hyperparameter combinations and save configs."""
        keys = grid.keys()
        values = grid.values()
        combinations = list(itertools.product(*values))
        
        print(f"Generated {len(combinations)} hyperparameter combinations")
        
        experiments = []
        for idx, combo in enumerate(combinations, start=1):
            exp_id = f"span_exp{idx:03d}"
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
    
    def run_experiment(self, exp_id, config_path, hyperparams):
        """Run train, infer, and evaluate for span extraction."""
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
            # Step 1: Training (5 models)
            print("\n[1/3] Training span extraction models (5 marker types)...")
            train_result = subprocess.run(
                ["python", "train_one_span.py"],
                capture_output=True,
                text=True,
                timeout=14400  # 4 hour timeout (5x longer than binary)
            )
            
            if train_result.returncode != 0:
                print(f"ERROR: Training failed for {exp_id}")
                print(train_result.stderr)
                return None
            
            print("✓ Training complete")
            
            # Step 2: Inference
            print("\n[2/3] Running inference...")
            infer_result = subprocess.run(
                ["python", "infer_one_span.py"],
                capture_output=True,
                text=True,
                timeout=1200  # 20 minute timeout
            )
            
            if infer_result.returncode != 0:
                print(f"ERROR: Inference failed for {exp_id}")
                print(infer_result.stderr)
                return None
            
            print("✓ Inference complete")
            
            # Step 3: Evaluation
            print("\n[3/3] Evaluating...")
            eval_result = subprocess.run(
                ["python", "eval_token.py",
                 "--ground_truth_file", "val_split.jsonl",
                 "--prediction_file", "submission.jsonl",
                 "--scores_output_file", "scores_span.json"],
                capture_output=True,
                text=True,
                timeout=240  # 4 minute timeout
            )
            
            if eval_result.returncode != 0:
                print(f"ERROR: Evaluation failed for {exp_id}")
                print(eval_result.stderr)
                return None
            
            print("✓ Evaluation complete")
            
            # Read scores
            with open("scores_span.json", 'r') as f:
                scores = json.load(f)
            
            training_time = time.time() - start_time
            
            # Prepare results
            results = {
                'exp_id': exp_id,
                'model_name': model_name,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'training_time_seconds': round(training_time, 2),
                'f1_aggregate': scores.get('F1_Aggregate_Token', 0.0),
                'f1_macro': scores.get('F1_Macro_Token', 0.0),
                'precision_aggregate': scores.get('Precision_Aggregate_Token', 0.0),
                'recall_aggregate': scores.get('Recall_Aggregate_Token', 0.0),
                'f1_action': scores.get('F1_Action_Token', 0.0),
                'f1_actor': scores.get('F1_Actor_Token', 0.0),
                'f1_effect': scores.get('F1_Effect_Token', 0.0),
                'f1_evidence': scores.get('F1_Evidence_Token', 0.0),
                'f1_victim': scores.get('F1_Victim_Token', 0.0),
            }
            
            # Add hyperparameters to results
            results.update(hyperparams)
            
            print(f"\n✓ Experiment {exp_id} complete!")
            print(f"  F1 Score (Aggregate): {results['f1_aggregate']:.4f}")
            print(f"  F1 Score (Macro): {results['f1_macro']:.4f}")
            print(f"  Training Time: {results['training_time_seconds']:.2f}s ({results['training_time_seconds']/60:.1f} min)")
            
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
        """Returns a set of experiment IDs that have already been completed."""
        if not os.path.exists(self.results_csv):
            return set()
        
        completed = set()
        with open(self.results_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row['exp_id'])
        
        return completed
    
    def run_grid_search(self):
        """Main method to run the complete grid search."""
        print("="*70)
        print("STARTING SPAN EXTRACTION GRID SEARCH (TASK 1)")
        print("="*70)
        
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
            print(f"Completed: {sorted(completed)[:10]}{'...' if len(completed) > 10 else ''}")
            experiments = [(exp_id, cfg, hp) for exp_id, cfg, hp in experiments if exp_id not in completed]
        
        print(f"\nTotal experiments to run: {len(experiments)}")
        
        if len(experiments) == 0:
            print("\n✓ All experiments already completed!")
            self.print_top_results()
            return
        
        # Estimate time
        avg_time_per_exp = 20  # minutes (rough estimate)
        total_time = len(experiments) * avg_time_per_exp
        print(f"\nEstimated time: ~{total_time} minutes ({total_time/60:.1f} hours)")
        print(f"Note: Span extraction trains 5 models per experiment (much slower than binary)")
        
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
        print("\n" + "="*70)
        print("SPAN EXTRACTION GRID SEARCH COMPLETE")
        print("="*70)
        print(f"Successful experiments: {successful_experiments}")
        print(f"Failed experiments: {failed_experiments}")
        print(f"\nResults saved to: {self.results_csv}")
        
        if successful_experiments > 0:
            self.print_top_results()
    
    def print_top_results(self, top_n=5):
        """Print top N results by F1 aggregate score."""
        if not os.path.exists(self.results_csv):
            return
        
        # Read results
        with open(self.results_csv, 'r') as f:
            reader = csv.DictReader(f)
            results = list(reader)
        
        if not results:
            return
        
        # Sort by F1 aggregate score
        results.sort(key=lambda x: float(x['f1_aggregate']), reverse=True)
        
        print(f"\n{'='*70}")
        print(f"TOP {min(top_n, len(results))} EXPERIMENTS BY F1 AGGREGATE SCORE")
        print(f"{'='*70}")
        
        for i, result in enumerate(results[:top_n], 1):
            print(f"\n{i}. {result['exp_id']} - F1 Agg: {result['f1_aggregate']} (Macro: {result['f1_macro']})")
            print(f"   Model: {result['model_name']}")
            print(f"   LR: {result['learning_rate']}, Batch: {result['batch_size']}, "
                  f"Epochs: {result['num_epochs']}, Warmup: {result['warmup_ratio']}")
            print(f"   Per-type F1s: Action={float(result['f1_action']):.3f}, "
                  f"Actor={float(result['f1_actor']):.3f}, "
                  f"Effect={float(result['f1_effect']):.3f}, "
                  f"Evidence={float(result['f1_evidence']):.3f}, "
                  f"Victim={float(result['f1_victim']):.3f}")


if __name__ == "__main__":
    print("Span Extraction Grid Search for Hyperparameter Tuning")
    print("=====================================================")
    print("\nNote: This trains 5 models per experiment (Action, Actor, Effect, Evidence, Victim)")
    print("Expect ~15-30 minutes per experiment depending on hyperparameters")
    
    runner = SpanGridSearchRunner()
    runner.run_grid_search()