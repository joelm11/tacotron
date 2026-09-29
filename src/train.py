"""Train the model."""

import os

# Keep TensorFlow as the Keras backend (must be set before importing keras)
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import pandas as pd

from data import get_metadata_path, get_project_root
from model import Tacotron

SPEECH_MD = get_metadata_path(get_project_root())


# Training loop
def train(batch_size=32):
    # Source training data
    df = pd.read_csv(SPEECH_MD, sep="|", header=None)
    # Batch training data
    # Push through model
    # Checkpoint
    return


def main():
    model = Tacotron()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer)
    # Compile?
    # Fit?
    train()


if __name__ == "__main__":
    main()
