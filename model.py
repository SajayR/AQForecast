import torch
import torch.nn as nn


'''Should accept inputs of (batch_size, seq_len, 1), and output (batch_size, pred_len, 1)'''

class Model(nn.Module):  #simple LSTM, replace with timefm or whichever model we need
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out)
        return out[:, -self.output_size:, :]


# model = Model(input_size=1, hidden_size=64, output_size=96, num_layers=2)