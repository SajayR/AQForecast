# Time Series Forecasting Framework

## Directory Structure

```
project_root/
├── data_config/
│   └── data_config_{identifier}.toml
├── exp_config/
│   └── exp_config_{identifier}.toml
├── models/
│   └── {model_name}/
│       ├── model.py
│       └── config_{identifier}.toml
├── results/
│   └── {model_name}/
│       └── {exp_config}/
│           ├── best_model.pth
│           ├── predictions.csv
│           └── metrics.json
├── utils/
│   ├── config.py
│   ├── data_loader.py
│   └── model.py
├── train.py
├── eval.py
├── main.py
└── compile_metrics.py
```

## Usage

### Training a Model

To train a model, use the following command:

```
python train.py --data-config {data_config_id} --exp-config {exp_config_id} --model {model_name} --model-config {model_config_id}
```

### Evaluating a Model

To evaluate a trained model, use:

```
python eval.py --data-config {data_config_id} --exp-config {exp_config_id} --model {model_name} --model-config {model_config_id}
```

### Training and Evaluating in One Go

To train and evaluate a model in one command, use:

```
python main.py --data-config {data_config_id} --exp-config {exp_config_id} --model {model_name} --model-config {model_config_id}
```

## Configuration Files

1. `data_config/{data_config_id}.toml`: Specifies dataset and dataloader parameters
2. `exp_config/{exp_config_id}.toml`: Specifies training and evaluation parameters
3. `models/{model_name}/config_{model_config_id}.toml`: Specifies model-specific parameters

## Adding a New Model

To add a new model, follow these steps:

1. Create a new directory under `models/` with your model name (e.g., `models/new_model/`).

2. In this directory, create a `model.py` file that implements the `BaseModel` class from `utils/model.py`. Your model class should have the following structure:

   ```python
   from utils.model import BaseModel, train_model, evaluate_model

   class Model(BaseModel):
       def __init__(self, seq_len, pred_len, ...):
           super(Model, self).__init__()
           # Initialize your model architecture here
           
       def forward(self, x):
           # Implement the forward pass of your model
           
       def fit(self, train_loader, val_loader, num_epochs, learning_rate, device, save_path):
           return train_model(self, train_loader, val_loader, num_epochs, learning_rate, device, save_path)
           
       def predict(self, x):
           self.eval()
           with torch.no_grad():
               return self(x)
   ```

3. Create a configuration file for your model (e.g., `config_default.toml`) in the same directory. This file should contain any model-specific parameters:

   ```toml
   [model]
   # Add your model-specific parameters here
   hidden_size = 64
   num_layers = 2
   # ... other parameters
   ```

4. Update the `data_config/{data_config_id}.toml` file if your model requires specific data preprocessing or different sequence/prediction lengths.

5. If necessary, update the `exp_config/{exp_config_id}.toml` file to include any specific training or evaluation settings for your model.


## Example Commands

Here are some example commands to help you get started:

1. To train and evaluate the LSTM model with default configurations:

   ```
   python main.py --data-config default --exp-config default --model lstm --model-config default
   ```

   This command will train the LSTM model, save the best model in the `results/lstm/default/` directory, and then evaluate it, saving the predictions and metrics in the same directory.

2. To train only the LSTM model:

   ```
   python train.py --data-config default --exp-config default --model lstm --model-config default
   ```

   This command will train the LSTM model and save the best model in the `results/lstm/default/` directory.

3. To evaluate a pre-trained LSTM model:

   ```
   python eval.py --data-config default --exp-config default --model lstm --model-config default
   ```

   This command will load a pre-trained LSTM model from `results/lstm/default/best_model.pth`, evaluate it on the test set, and save the predictions and metrics in the same directory.

Note: These commands assume you're using the default configuration files. If you've created custom configurations, replace `default` with your configuration identifiers as needed.

