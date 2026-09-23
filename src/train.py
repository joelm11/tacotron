"""Train the model."""

from pathlib import Path

import pandas as pd

# TODO: Un-hardcode this.
SPEECH_MD = Path("/home/joelm/Desktop/tts/taco/data/raw") / "LJSpeech-1.1" / "metadata.csv"

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
