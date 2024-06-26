# models/lstm/model.py

import torch
import torch.nn as nn
from utils.model import BaseModel, train_model, evaluate_model

class Model(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.lstm = nn.LSTM(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            batch_first=True
        )
        self.fc = nn.Linear(config['model']['hidden_size'], config['training']['pred_len'])

    def forward(self, x):
        h0 = torch.zeros(self.config['model']['num_layers'], x.size(0), self.config['model']['hidden_size']).to(x.device)
        c0 = torch.zeros(self.config['model']['num_layers'], x.size(0), self.config['model']['hidden_size']).to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out.unsqueeze(-1)

    def fit(self, train_loader):
        device = torch.device(self.config['training']['device'])
        self.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['optimizer']['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        return train_model(
            self, train_loader, criterion, optimizer, scheduler,
            self.config['optimizer']['num_epochs'], device, self.config['saving']['save_path']
        )

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)