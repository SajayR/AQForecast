import torch
import torch.nn as nn
from pathlib import Path
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def fit(self, train_loader, val_loader, num_epochs, learning_rate, device):
        pass

    @abstractmethod
    def predict(self, x):
        pass


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_path):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    best_loss = float('inf')
    
    # Ensure the save_path directory exists
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x_enc, y in train_loader:
            x_enc = x_enc.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            output = model(x_enc)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{save_path}/best_model.pth")
            print(f"Saved best model with loss: {best_loss:.4f}")

    return best_loss

# ... (keep the existing evaluate_model function)

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_enc, y in data_loader:
            x_enc = x_enc.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            output = model(x_enc)
            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / len(data_loader)