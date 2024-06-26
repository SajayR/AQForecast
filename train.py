import torch
from pathlib import Path
from utils.data_loader import load_data, create_dataloaders
import importlib

def train(config):
    data = load_data(config['data']['csv_file'], config['data']['target_column'])
    train_loader, _ = create_dataloaders(
        data,
        config['training']['seq_len'],
        config['training']['pred_len'],
        config['model']['batch_size'],
        config['data']['num_workers']
    )

    model_module = importlib.import_module(f"models.{config['model']['type']}.model")
    Model = getattr(model_module, "Model")
    
    model = Model(config)
    
    device = torch.device(config['training']['device'])
    model.to(device)
    
    best_loss = model.fit(train_loader)
    
    print(f"Training completed. Best model saved with loss: {best_loss:.4f}")