import torch
import pandas as pd
import json
from pathlib import Path
from utils.data_loader import load_data, create_dataloaders
import importlib

def evaluate(config):
    data = load_data(config['data']['csv_file'], config['data']['target_column'])
    _, test_loader = create_dataloaders(
        data,
        config['training']['seq_len'],
        config['training']['pred_len'],
        config['model']['batch_size'],
        config['data']['num_workers']
    )


    model_module = importlib.import_module(f"models.{config['model']['type']}.model")
    Model = getattr(model_module, "Model")
    
    model = Model(config)
    model.load(f"{config['saving']['save_path']}/best_model.pt")
    
    device = torch.device(config['training']['device'])
    model.to(device)
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for x_enc, y in test_loader:
            x_enc = x_enc.to(device)
            output = model.predict(x_enc)
            predictions.extend(output.cpu().numpy().reshape(-1).tolist())
            targets.extend(y.numpy().reshape(-1).tolist())
    
    df = pd.DataFrame({'predictions': predictions, 'targets': targets})
    
    mse = ((df['predictions'] - df['targets']) ** 2).mean()
    mae = (df['predictions'] - df['targets']).abs().mean()
    
    metrics = {
        'mse': mse,
        'mae': mae
    }
    
    save_path = Path(config['saving']['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(save_path / "predictions.csv", index=False)
    with open(save_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Evaluation completed. MSE: {mse:.4f}, MAE: {mae:.4f}")
    print(f"Results and metrics saved in {save_path}")

if __name__ == "__main__":
    import argparse
    from utils.config import load_config

    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate(config)