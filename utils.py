import argparse
import torch
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class TestDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return (len(self.data) - self.seq_len - self.pred_len + 1) // 1000 + 1

    def __getitem__(self, index):
        start_idx = index * 1000
        x_enc = self.data[start_idx:start_idx + self.seq_len]
        y = self.data[start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len]
        return x_enc, y

class TrainDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = self.seq_len + self.pred_len
        self.valid_indices = self._get_valid_indices()

    def _get_valid_indices(self):
        valid_indices = []
        for i in range(len(self.data) - self.total_len + 1):
            start_block = i // 1000
            end_block = (i + self.total_len - 1) // 1000
            if start_block == end_block and i % 1000 >= self.total_len:
                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        start_idx = self.valid_indices[index]
        x_enc = self.data[start_idx:start_idx + self.seq_len]
        y = self.data[start_idx + self.seq_len:start_idx + self.total_len]
        return x_enc, y

def load_data(csv_file, target_column):
    data = pd.read_csv(csv_file)
    target_data = data[target_column].values.reshape(-1, 1)
    return target_data

def create_dataloaders(data, seq_len, pred_len, batch_size):
    train_dataset = TrainDataset(data, seq_len, pred_len)
    test_dataset = TestDataset(data, seq_len, pred_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, num_epochs, learning_rate, device, save_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x_enc, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x_enc = x_enc.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            x_enc = x_enc.view(x_enc.size(0), -1, 1)
            y = y.view(y.size(0), -1, 1)
            
            optimizer.zero_grad()
            output = model(x_enc)
            print(x_enc.shape)
            print(output.shape)
            print(y.shape)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        test_loss, test_mae = evaluate_model(model, test_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')
        
        scheduler.step(test_loss)
        
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f"Saved best model with loss: {best_loss:.4f}")

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_mae = 0
    with torch.no_grad():
        for x_enc, y in tqdm(test_loader, desc="Evaluating"):
            x_enc = x_enc.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            x_enc = x_enc.view(x_enc.size(0), -1, 1)
            y = y.view(y.size(0), -1, 1)
            output = model(x_enc)
            test_loss += criterion(output, y).item()
            test_mae += torch.mean(torch.abs(output - y)).item()
    return test_loss / len(test_loader), test_mae / len(test_loader)

def main(args, model_class, model_args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = load_data(args.csv_file, args.target_column)
    train_loader, test_loader = create_dataloaders(data, args.seq_len, args.pred_len, args.batch_size)
    
    model = model_class(**model_args).to(device)
    print(model)
    
    os.makedirs(args.save_path, exist_ok=True)
    
    train_model(model, train_loader, test_loader, args.num_epochs, args.learning_rate, device, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting Training Script")
    parser.add_argument('--csv_file', type=str, default='univariate_interpolated_IGI_airport.csv', help='Path to the CSV file')
    parser.add_argument('--target_column', type=str, default='PM2.5 (µg/m³)', help='Target column name in the CSV')
    parser.add_argument('--seq_len', type=int, default=96, help='Sequence length for input')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_class', type=type, required=True, help='Model class to use')
    parser.add_argument('--model_args', nargs='*', help='Arguments to pass to the model constructor')
    
    args = parser.parse_args()
    main(args)