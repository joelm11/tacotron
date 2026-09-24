"""Train the model."""

import pandas as pd

from data import get_metadata_path, get_project_root

SPEECH_MD = get_metadata_path(get_project_root())

# Get input sequence and acoustic representations


# Training loop
def train(batch_size=32):
    # Source training data
    df = pd.read_csv(SPEECH_MD, sep="|", header=None)
    # Batch training data
    # Push through model
    # Checkpoint
    return


def main():
    train()


if __name__ == "__main__":
    main()
