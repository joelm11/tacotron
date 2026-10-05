# Data pre-processing methods

import csv
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from data import get_metadata_path, get_project_root, get_wav_path, setup_dir_structure

# Construct batches from feature metadata and offline features (bucketing)

MAX_TEXT_LEN = 256
MAX_FRAMES_LEN = 10
TARGET_SR = 24e3
N_MELS = 80


@dataclass
class Sample:
    label: str
    text: str
    wav_path: Path


def text_embeddings(text: str):
    ints = [ord(ch) for ch in text[:MAX_TEXT_LEN]]
    ints_pad = ints + [0] * (MAX_TEXT_LEN - len(ints))
    ohe = np.eye(256, dtype=np.float32)[ints_pad]
    return ohe


def mel_scale_spectrogram(y, sr):
    frame_size = int(sr * 0.05)
    hop_size = int(sr * 0.0125)
    mels_db = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        win_length=frame_size,
        hop_length=hop_size,
        center=False,
        n_mels=N_MELS,
        window="hann",
    )
    return librosa.power_to_db(mels_db, ref=1.0).T.astype(np.float32)


def linear_scale_spectrogram(y, sr):
    frame_size = int(sr * 0.05)
    hop_size = int(sr * 0.0125)
    stft = librosa.stft(
        y=y,
        n_fft=2048,
        win_length=frame_size,
        hop_length=hop_size,
        window="hann",
        center=False,
    )
    mag = np.abs(stft)
    return librosa.amplitude_to_db(mag, ref=1.0).T.astype(np.float32)


def pre_proc_data():
    # Generates features from raw data. Writes manifest on processed data.
    p_root = get_project_root()
    _, f_out_dir = setup_dir_structure(p_root)

    # Parse sample metadata
    samples = []
    with open(get_metadata_path(p_root), newline="") as f:
        md = csv.reader(f, delimiter="|")
        for row in md:
            # LJSpeech: label | text | normalized_text
            label, text = row[0], row[2] if len(row) > 2 else row[1]
            samples.append(Sample(label, text, get_wav_path(p_root, label)))

    # Extract features from samples
    procd_samples = []
    with Pool() as pool:
        procd_samples = list(
            tqdm(
                pool.imap_unordered(partial(_proc_sample, out_dir=f_out_dir), samples),
                total=len(samples),
            )
        )
    valid_samples = [sample for sample in procd_samples if sample is not None]

    # Write manifest
    manifest_path = f_out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["label", "path", "text_len", "mel_len"])
        for feat_path in tqdm(valid_samples, desc="Writing manifest"):
            try:
                data = np.load(feat_path)
                text_len = data["text"].shape[0]
                mel_len = data["mels"].shape[0]
                writer.writerow(
                    [
                        feat_path.stem,
                        str(feat_path),
                        text_len,
                        mel_len,
                    ]
                )
                data.close()
            except Exception as e:
                print(f"Failed to read {feat_path}, skipping: {e}")

    return manifest_path


def _proc_sample(sample, out_dir):
    out_path = out_dir / f"{sample.label}.npz"
    if out_path.exists():
        return out_path

    try:
        y, sr = librosa.load(sample.wav_path, sr=TARGET_SR)

        # Compute features
        ohe_text = text_embeddings(sample.text)
        mels = mel_scale_spectrogram(y, sr)
        spectra = linear_scale_spectrogram(y, sr)

        np.savez_compressed(out_path, text=ohe_text, mels=mels, spectra=spectra)
        return out_path

    except Exception:
        print(f"Failed to process {sample.label}")
        return None


def make_dataset(batch_size=32, r=5):
    # Bucket sample data based on text len by reading manifest
    # Text in a bucket is padded to the max text length in the bucket
    # Mels and spectra are padded to the max mel length in the bucket that is divisible by r
    # Construct batches from buckets
    return


def main():
    pre_proc_data()


if __name__ == "__main__":
    main()
