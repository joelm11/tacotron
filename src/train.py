"""Train the model."""

import pandas as pd

from data import get_metadata_path, get_project_root

SPEECH_MD = get_metadata_path(get_project_root())

# Get input sequence and acoustic representations


# Training loop
def train(batch_size=32):
    # Get entry from metadata
    df = pd.read_csv(SPEECH_MD, sep="|", header=None)
    for row in df.itertuples():
        print(row[2])
        break


def main():
    train()


if __name__ == "__main__":
    main()
