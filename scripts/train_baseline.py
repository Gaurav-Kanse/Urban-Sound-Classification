"""Train the fold-based Random Forest baseline."""

import json

from soundsense.training.train import train_baseline

if __name__ == "__main__":
    print(json.dumps(train_baseline(), indent=2))
