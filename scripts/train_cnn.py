"""Train the log-Mel CNN."""

import argparse
import json

from soundsense.training.train import train_cnn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(train_cnn(args.epochs, args.batch_size, args.patience), indent=2))
