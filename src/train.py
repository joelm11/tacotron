"""Train the model."""

import os

from preproc import build_dataset

# Keep TensorFlow as the Keras backend (must be set before importing keras)
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras

from data import get_metadata_path, get_project_root
from model import Tacotron

SPEECH_MD = get_metadata_path(get_project_root())


# # Training loop
# def train(model, batch_size=32):
#     # Source training data
#     train_data = build_dataset(batch_size)

#     # Print the dataset's static shape signature
#     print("Dataset element_spec:", train_data.element_spec)

#     # Print the actual shapes of one concrete batch
#     for text, mels, spectra in train_data.take(1):
#         print("Batch shapes -- text:", text.shape, "mels:", mels.shape, "spectra:", spectra.shape)

#     return


def main():
    model = Tacotron()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer,
        loss={"mels": "mae", "spectra": "mae"},
        metrics={"mels": "mae", "spectra": "mae"},
    )
    # model.build()
    model.fit(build_dataset(32))


if __name__ == "__main__":
    main()
