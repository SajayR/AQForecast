import pandas as pd
import torch
from pathlib import Path

def create_dataset(csv_file, target_column, save_path):
    data = pd.read_csv(csv_file)
    target_data = data[target_column].values.reshape(-1, 1)
    
    # Convert to PyTorch tensor
    tensor_data = torch.FloatTensor(target_data)
    
    # Ensure the save_path directory exists
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Save the tensor
    torch.save(tensor_data, f"{save_path}/dataset.pt")
    
    print(f"Dataset saved to {save_path}/dataset.pt")

if __name__ == "__main__":
    csv_file = "../data/IGI_airport.csv"
    target_column = "PM2.5 (µg/m³)"
    save_path = "../data/processed"
    
    create_dataset(csv_file, target_column, save_path)