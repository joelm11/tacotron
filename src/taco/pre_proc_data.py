# Data pre-processing methods

import csv
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from taco.data import (
    get_feat_dir,
    get_metadata_path,
    get_project_root,
    get_wav_path,
    setup_dir_structure,
)

SR = 22050
N_FFT = 2048
FRAME_SHIFT = 0.0125  # seconds
FRAME_LENGTH = 0.05  # seconds
HOP_LENGTH = int(SR * FRAME_SHIFT)  # 275
WIN_LENGTH = int(SR * FRAME_LENGTH)  # 1102
N_MELS = 80
PREEMPHASIS = 0.97
MAX_DB = 100
REF_DB = 20
MAX_TEXT_LEN = 256


@dataclass
class Sample:
    label: str
    text: str
    wav_path: Path


def text_embeddings(text: str):
    ints = [ord(ch) for ch in text[:MAX_TEXT_LEN]]
    ohe = np.eye(256, dtype=np.float32)[ints]
    return ohe


def _get_spectrograms(y, sr):
    """Returns normalized log-melspectrogram and log-magnitude spectrogram."""
    # Stft
    linear = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # Mel spectrogram
    mel_basis = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, T)

    # Db
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # Normalize
    mel = np.clip((mel - REF_DB + MAX_DB) / MAX_DB, 1e-8, 1)
    mag = np.clip((mag - REF_DB + MAX_DB) / MAX_DB, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


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
        y, sr = librosa.load(sample.wav_path, sr=SR)

        # Trimming
        y, _ = librosa.effects.trim(y)

        # Preemphasis
        y = np.append(y[0], y[1:] - PREEMPHASIS * y[:-1])

        # Compute features
        ohe_text = text_embeddings(sample.text)
        mels, spectra = _get_spectrograms(y, sr)

        np.savez_compressed(out_path, text=ohe_text, mels=mels, spectra=spectra)
        return out_path

    except Exception:
        print(f"Failed to process {sample.label}")
        return None


def make_dataset(batch_size=32, r=5):
    """Bucketed tf.data.Dataset from pre-processed features."""
    feat_dir = get_feat_dir(get_project_root())
    rows = list(csv.DictReader(open(feat_dir / "manifest.csv")))
    buckets = {}
    for row in rows:
        buckets.setdefault(max(int(row["text_len"]) - 1, 0) // 16, []).append(row)

    def load(p):
        with np.load(p.decode()) as d:
            return (
                d["text"].astype(np.float32),
                d["mels"].astype(np.float32),
                d["spectra"].astype(np.float32),
            )

    A = tf.data.AUTOTUNE
    dss, w = [], []
    for k in sorted(buckets):
        b = buckets[k]
        if len(b) < batch_size:
            continue
        max_t = max(int(r["text_len"]) for r in b)
        max_m = ((max(int(r["mel_len"]) for r in b) + r - 1) // r) * r
        paths = [r["path"] for r in b]
        pad = lambda x, l: tf.pad(x[:l], [[0, l - tf.shape(x)[0]], [0, 0]])

        def proc(t, m, s, mt=max_t, mm=max_m):
            t.set_shape([None, 256])
            m.set_shape([None, 80])
            s.set_shape([None, 1025])
            return pad(t, mt), pad(m, mm), pad(s, mm)

        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = (
            ds.shuffle(len(paths))
            .map(lambda p: tf.numpy_function(load, [p], 3 * [tf.float32]), A)
            .map(proc, A)
            .padded_batch(batch_size, drop_remainder=True)
            .map(lambda t, m, s: ({"text": t, "mels": m}, {"pred_mels": m, "pred_spectra": s}), A)
            .prefetch(A)
        )
        dss.append(ds)
        w.append(float(len(b)))

    return (
        tf.data.Dataset.sample_from_datasets(dss, w)
        if dss
        else tf.data.Dataset.from_tensor_slices([])
    )


def main():
    pre_proc_data()


if __name__ == "__main__":
    main()
