"""Train the model."""

import os

import gpu_setup  # noqa: F401  -- preload NVIDIA libs before TF/Keras

# Keep TensorFlow as the Keras backend (must be set before importing keras)
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf

from data import get_metadata_path, get_project_root
from model1 import TextToSpectra
from preproc import build_dataset

SPEECH_MD = get_metadata_path(get_project_root())


def main():
    # Allow TF to grow GPU memory rather than allocate all at once
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    BATCH_SIZE = 32
    model = TextToSpectra()
    model.compile(
        _build_optimizer(),
        loss={"pred_mels": "mae", "pred_spectra": "mae"},
        metrics={"pred_mels": "mae", "pred_spectra": "mae"},
    )

    ds = build_dataset(BATCH_SIZE)
    model.fit(ds, epochs=10)


def _build_optimizer():
    boundaries = [500e3, 1e6, 2e6]
    values = [0.001, 0.0005, 0.0003, 0.0001]

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries, values=values
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return optimizer


if __name__ == "__main__":
    main()
