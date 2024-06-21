import argparse
import json
from pathlib import Path
import pandas as pd

def compile_metrics(results_dir, save_path):
    results = []
    for model_path in Path(results_dir).glob("*"):
        model_name = model_path.name
        for exp_config_path in model_path.glob("*"):
            exp_config = exp_config_path.name
            metrics_file = exp_config_path / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                result = {
                    'model': model_name,
                    'exp_config': exp_config,
                }
                result.update(metrics)
                results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Compiled metrics saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile metrics from all experiments")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing results")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save compiled metrics")
    
    args = parser.parse_args()
    compile_metrics(args.results_dir, args.save_path)