from model import Model
from utils import main 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting Training Script")
    parser.add_argument('--csv_file', type=str, default='data/IGI_airport.csv', help='Path to the CSV file')
    parser.add_argument('--target_column', type=str, default='PM2.5 (µg/m³)', help='Target column name in the CSV')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length for input')
    parser.add_argument('--pred_len', type=int, default=12, help='Prediction length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size of the LSTM')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--save_path', type=str, default='model_checkpoints', help='Path to save model checkpoints')
    
    args = parser.parse_args()
    
    # Model class and arguments
    model_args = {
        'input_size': 1,
        'hidden_size': args.hidden_size,
        'output_size': args.pred_len,
        'num_layers': args.num_layers
    }
    
    main(args, Model, model_args)