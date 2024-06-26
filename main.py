import argparse
from train import train
from eval import evaluate
from utils.config import load_config

def main(config):
    print("Starting training...")
    train(config)
    print("Training completed. Starting evaluation...")
    evaluate(config)
    print("Evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)