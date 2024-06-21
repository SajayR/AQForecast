import argparse
import torch
import json
import pandas as pd
from pathlib import Path
from utils.config import load_data_config, load_exp_config, load_model_config
from utils.data_loader import load_data, create_dataloaders
from utils.model import evaluate_model
import importlib

def evaluate(args):
    data_config = load_data_config(args.data_config)
    exp_config = load_exp_config(args.exp_config)
    model_config = load_model_config(args.model, args.model_config)

    data = load_data(data_config['dataset']['csv_file'], data_config['dataset']['target_column'])
    _, test_loader = create_dataloaders(
        data, 
        data_config['dataset']['seq_len'],
        data_config['dataset']['pred_len'],
        data_config['dataloader']['batch_size'],
        data_config['dataloader']['num_workers']
    )

    device = torch.device(exp_config['training']['device'])

    # Dynamically import the model
    model_module = importlib.import_module(f"models.{args.model}.model")
    Model = getattr(model_module, "Model")
    
    # Update model_config with sequence and prediction lengths
    model_config['model'].update({
        'seq_len': data_config['dataset']['seq_len'],
        'pred_len': data_config['dataset']['pred_len']
    })
    
    model = Model(**model_config['model']).to(device)
    
    save_path = Path(f"results/{args.model}/{args.exp_config}")
    model.load_state_dict(torch.load(f"{save_path}/best_model.pth"))
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for x_enc, y in test_loader:
            x_enc = x_enc.to(device, dtype=torch.float32)
            output = model.predict(x_enc)
            predictions.extend(output.cpu().numpy().reshape(-1).tolist())
            targets.extend(y.numpy().reshape(-1).tolist())
    
    df = pd.DataFrame({'predictions': predictions, 'targets': targets})
    df.to_csv(f"{save_path}/predictions.csv", index=False)
    
    metrics = {}
    for metric in exp_config['evaluation']['metrics']:
        if metric == 'mse':
            metrics['mse'] = ((df['predictions'] - df['targets']) ** 2).mean()
        elif metric == 'mae':
            metrics['mae'] = (df['predictions'] - df['targets']).abs().mean()
    
    with open(f"{save_path}/metrics.json", 'w') as f:
        json.dump(metrics, f)

    print(f"Evaluation completed. Results saved in {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--data-config", type=str, required=True, help="Data configuration identifier")
    parser.add_argument("--exp-config", type=str, required=True, help="Experiment configuration identifier")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--model-config", type=str, required=True, help="Model configuration identifier")
    
    args = parser.parse_args()
    evaluate(args)