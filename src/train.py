"""Train the model."""

import os

from data import get_project_root
from pre_proc_data import make_dataset
from taco import Tts

# Keep TensorFlow as the Keras backend (must be set before importing keras)
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf


def overfit():
    # Allow TF to grow GPU memory rather than allocate all at once
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    model = Tts()
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={"pred_mels": "mae", "pred_spectra": "mae"},
    )

    # Grab 1 batch of size 1 and materialize it so we overfit the same sample
    ds = make_dataset(1, 5)
    for x_batch, y_batch in ds.take(1):
        pass

    single_batch_ds = tf.data.Dataset.from_tensors((x_batch, y_batch)).repeat()  # type: ignore
    model.fit(
        single_batch_ds,
        steps_per_epoch=1,  # 1 step per epoch ensures it only sees this exact batch
        epochs=1000,  # Enough steps to show clear convergence on a single sample
    )


def train():
    # Allow TF to grow GPU memory rather than allocate all at once
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    model = Tts()
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={"pred_mels": "mae", "pred_spectra": "mae"},
    )
    model.fit(
        make_dataset(),
        steps_per_epoch=10,
        epochs=100,
    )
    model.save(get_project_root() / "model.keras")


def main():
    train()
    # overfit()


if __name__ == "__main__":
    main()
