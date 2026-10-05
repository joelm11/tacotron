"""Train the model."""

import os

from pre_proc_data import make_dataset
from taco import Tts

# Keep TensorFlow as the Keras backend (must be set before importing keras)
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf

from data import get_metadata_path, get_project_root

SPEECH_MD = get_metadata_path(get_project_root())


def main():
    # Allow TF to grow GPU memory rather than allocate all at once
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    BATCH_SIZE = 32
    model = Tts()
    model.compile(
        _build_optimizer(),
        loss={"pred_mels": "mae", "pred_spectra": "mae"},
        metrics={"pred_mels": "mae", "pred_spectra": "mae"},
    )

    ds = make_dataset(4, 5)
    # Grab 1 batch of size 4 and repeat it infinitely for fit()
    single_batch_ds = ds.take(1).repeat()

    model.fit(
        single_batch_ds,
        # make_dataset(32, 5),
        steps_per_epoch=1,  # 1 step per epoch ensures it only sees this exact batch
        epochs=1000,  # Tacotron models often need 200-1000+ steps to collapse loss
    )


# model.fit(ds, epochs=10)


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
