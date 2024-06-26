# Time Series Forecasting Framework

## Project Structure

```
project_root/
├── data/
│   └── IGI_airport.csv
├── models/
│   └── lstm/
│       ├── model.py
│       └── config_default.toml
├── utils/
│   ├── config.py
│   ├── data_loader.py
│   └── model.py
├── train.py
├── eval.py
├── main.py
└── README.md
```

## Usage

### Training and Evaluating a Model

To train and evaluate a model, use the following command:

```
python main.py --config models/lstm/config_default.toml
```

This will train the model using the specifications in the config file, save the best model, and then evaluate it on the test set.

### Evaluating a Pre-trained Model

To evaluate a pre-trained model without retraining, use:

```
python eval.py --config models/lstm/config_default.toml
```

## Configuration

The `models/lstm/config_default.toml` file contains all the necessary configurations for the data, model, training, and evaluation processes. Here's a brief overview of the configuration sections:

- `[data]`: Specifies the data file and target column
- `[model]`: Defines the model architecture
- `[training]`: Sets training parameters like sequence length and device
- `[optimizer]`: Configures the optimizer settings
- `[saving]`: Specifies where to save the model and results

Modify this file to change hyperparameters or experiment settings.

## Adding New Models

To add a new model:

1. Create a new directory under `models/` with your model name (e.g., `models/gru/`).
2. Implement your model in a `model.py` file in this new directory. Your model should inherit from the `BaseModel` class in `utils/model.py`.
3. Create a new configuration file (e.g., `config_default.toml`) in your model's directory.
4. Update the `models/__init__.py` file to include your new model.

## Results

After training and evaluation, you can find the following in the save path specified in your config file:

- `best_model.pt`: The saved model with the best performance
- `predictions.csv`: CSV file containing the model's predictions and actual values
- `metrics.json`: JSON file containing evaluation metrics (MSE and MAE)
