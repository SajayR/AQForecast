import argparse
from train import train
from eval import evaluate

def main(args):
    print("Starting training...")
    train(args)
    print("Training completed. Starting evaluation...")
    evaluate(args)
    print("Evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model")
    parser.add_argument("--data-config", type=str, required=True, help="Data configuration identifier")
    parser.add_argument("--exp-config", type=str, required=True, help="Experiment configuration identifier")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--model-config", type=str, required=True, help="Model configuration identifier")
    
    args = parser.parse_args()
    main(args)