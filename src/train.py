"""Train the model."""

import argparse
import os
import sys

# Keep TensorFlow as the Keras backend (must be set before importing keras)
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import pandas as pd
import tensorflow as tf

from data import get_metadata_path, get_project_root
from model import Tacotron

SPEECH_MD = get_metadata_path(get_project_root())


def log_devices():
    backend = keras.config.backend()
    print(f"Keras backend: {backend}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {gpus}")
    if not gpus:
        print("WARNING: No GPU detected. Training will run on CPU.")


# Training loop
def train(batch_size=32):
    # Source training data
    df = pd.read_csv(SPEECH_MD, sep="|", header=None)
    # Batch training data
    # Push through model
    # Checkpoint
    return


def main():
    parser = argparse.ArgumentParser(description="Train the Tacotron model")
    parser.add_argument("--gpu", action="store_true", help="Require GPU for training")
    args = parser.parse_args()

    log_devices()

    if args.gpu and not tf.config.list_physical_devices("GPU"):
        print("ERROR: --gpu requested but no GPU is available.")
        sys.exit(1)

    model = Tacotron()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer)
    # Compile?
    # Fit?
    train()


if __name__ == "__main__":
    main()
