import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def fit(self, train_loader):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def save(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device, save_path):
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x_enc, y in train_loader:
            x_enc = x_enc.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            output = model(x_enc)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')
        
        scheduler.step(train_loss)
        
        if train_loss < best_loss:
            best_loss = train_loss
            model.save(f"{save_path}/best_model.pt")
            print(f"Saved best model with loss: {best_loss:.4f}")

    return best_loss

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_enc, y in data_loader:
            x_enc = x_enc.to(device)
            y = y.to(device)
            output = model(x_enc)
            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / len(data_loader)