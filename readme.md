# PM2.5 Forecasting Boilerplate

## Structure

- `main.py`: Entry point for training
- `model.py`: Model definition
- `utils.py`: Utility funcs for data handling and training

## Quick Start

1. `pip install -r requirements.txt`
2. Run training:
   ```
   python main.py --csv_file data/IGI_airport.csv --target_column 'PM2.5 (µg/m³)' --seq_len 128 --pred_len 12
   ```
   
## Customization

To add new models, edit the model.py folder and run the training. Make sure the the model accepts inputs of shape (batch_size, seq_len, 1) and outputs (batch_size, pred_len, 1) and you shouldn't need to edit the main.py file.

