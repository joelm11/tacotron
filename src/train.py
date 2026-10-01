"""Train the model."""

import os

from model1 import TextToSpectra

# Keep TensorFlow as the Keras backend (must be set before importing keras)
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import tensorflow as tf

from data import get_metadata_path, get_project_root
from preproc import build_dataset

SPEECH_MD = get_metadata_path(get_project_root())


def main():
    # Allow TF to grow GPU memory rather than allocate all at once
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    BATCH_SIZE = 1

    model = TextToSpectra()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer,
        loss={"pred_mels": "mae", "pred_spectra": "mae"},
        metrics={"pred_mels": "mae", "pred_spectra": "mae"},
    )

    ds = build_dataset(BATCH_SIZE)
    model.fit(ds, epochs=1)


if __name__ == "__main__":
    main()
