# Data pre-processing methods

import librosa
import numpy as np

from taco.hparams import DataConfig


def pre_proc_audio(audio, sr):
    return compute_mel_spectrogram(audio, sr)


# Mels as described in T2 paper
def compute_mel_spectrogram(audio, sr):
    frame_length = int(DataConfig.frame_ms * 1e-3 * sr)
    hop_length = int(DataConfig.frame_hop_ms * 1e-3 * sr)
    n_fft = DataConfig.n_fft

    audio, _ = librosa.effects.trim(audio)

    stft = librosa.stft(
        y=audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=frame_length,
        window="hann",
        center=True,
    )

    magnitude = np.abs(stft)

    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=DataConfig.n_mels,
        fmin=DataConfig.f_min,
        fmax=DataConfig.f_max,
        htk=False,
    )

    mel_spectrogram = np.dot(mel_basis, magnitude)

    mel_clipped = np.clip(mel_spectrogram, a_min=0.01, a_max=None)

    log_mel_spectrogram = np.log(mel_clipped)

    return log_mel_spectrogram
