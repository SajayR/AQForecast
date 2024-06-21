import argparse
import torch
from pathlib import Path
from utils.config import load_data_config, load_exp_config, load_model_config
from utils.data_loader import load_data, create_dataloaders
import importlib


def train(args):
    data_config = load_data_config(args.data_config)
    exp_config = load_exp_config(args.exp_config)
    model_config = load_model_config(args.model, args.model_config)

    data = load_data(data_config['dataset']['csv_file'], data_config['dataset']['target_column'])
    train_loader, val_loader = create_dataloaders(
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
    save_path.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_loader,
        val_loader,
        exp_config['training']['num_epochs'],
        exp_config['training']['learning_rate'],
        device,
        save_path
    )

    print(f"Training completed. Model saved at {save_path}/best_model.pth")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--data-config", type=str, required=True, help="Data configuration identifier")
    parser.add_argument("--exp-config", type=str, required=True, help="Experiment configuration identifier")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--model-config", type=str, required=True, help="Model configuration identifier")
    
    args = parser.parse_args()
    train(args)