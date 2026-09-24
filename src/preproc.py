"""Preprocessing for input text and training data."""

import keras
import librosa
import numpy as np

from data import get_project_root, get_wav_path

MAX_SEQ_LENGTH = 256
MAX_WAV_LEN_S = 10


# Generate character embeddings for an input string
def text_embeddings(text: str):
    # Convert characters to ints, pad the sequence, then ohe.
    ints = [ord(ch) for ch in text[:MAX_SEQ_LENGTH]]
    ints_pad = ints + [0] * (MAX_SEQ_LENGTH - len(ints))
    ohe = keras.utils.to_categorical(ints_pad, num_classes=256)
    return ohe


# Generate mels for a given training sample
def mel_scale_spectrogram(wav_file):
    y, sr = librosa.load(wav_file)

    # Pad input
    pad_samples = MAX_WAV_LEN_S * sr - y.shape[0]
    y = np.pad(y, (0, pad_samples))

    # Frame size and hop size, n_fft as described by paper
    frame_size = sr * 0.05
    hop_size = sr * 0.0125
    n_mels = 80
    mels_db = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        win_length=int(frame_size),
        hop_length=int(hop_size),
        center=False,
        n_mels=n_mels,
        window="hann",
    )
    return librosa.power_to_db(mels_db, ref=np.max)


# Generate full spectrogram for a given training sample
def linear_scale_spectrogram(wav_file):
    y, sr = librosa.load(wav_file)

    # Pad input
    pad_samples = MAX_WAV_LEN_S * sr - y.shape[0]
    y = np.pad(y, (0, pad_samples))

    # Frame size and hop size, n_fft as described by paper
    frame_size = sr * 0.05
    hop_size = sr * 0.0125
    stft = librosa.stft(
        y=y,
        n_fft=2048,
        win_length=int(frame_size),
        hop_length=int(hop_size),
        window="hann",
    )
    return librosa.power_to_db(stft, ref=np.max)


# Generate character embeddings and acoustic features from a given training data label
def preproc_entry(label: str, text: str):
    wav_file = get_wav_path(get_project_root(), label)
    return (
        text_embeddings(text),
        mel_scale_spectrogram(wav_file),
        linear_scale_spectrogram(wav_file),
    )


def main():
    pass


if __name__ == "__main__":
    main()
