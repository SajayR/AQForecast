import torch
import torch.nn as nn
from utils.model import BaseModel, train_model, evaluate_model

class Model(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, seq_len, pred_len):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out.unsqueeze(-1)
    
    def fit(self, train_loader, val_loader, num_epochs, learning_rate, device, save_path):
        return train_model(self, train_loader, val_loader, num_epochs, learning_rate, device, save_path)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)