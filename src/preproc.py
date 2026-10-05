"""Preprocessing for input text and training data.
We want to preprocess our features as they are constant and expensive to compute (mels and spectra).
We will serialize the computed features as .npz.
"""

import csv
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data import (
    get_feat_dir,
    get_metadata_path,
    get_project_root,
    get_wav_path,
    setup_dir_structure,
)

MAX_SEQ_LENGTH = 256
MAX_WAV_LEN_S = 10
TARGET_SR = 24e3
N_MELS = 80
N_FRAMES = 800


@dataclass
class MetadataEntry:
    label: str
    text: str
    wav_path: Path


def text_embeddings(text: str):
    ints = [ord(ch) for ch in text[:MAX_SEQ_LENGTH]]
    ints_pad = ints + [0] * (MAX_SEQ_LENGTH - len(ints))
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


def preproc_entry(entry: MetadataEntry, out_dir):
    out_path = out_dir / f"{entry.label}.npz"
    if out_path.exists():
        return out_path

    try:
        y, sr = librosa.load(entry.wav_path, sr=TARGET_SR)

        # Compute features
        ohe_text = text_embeddings(entry.text)
        mels = mel_scale_spectrogram(y, sr)
        spectra = linear_scale_spectrogram(y, sr)

        np.savez_compressed(
            out_path,
            text=ohe_text,
            mels=mels,
            spectra=spectra,
        )
    except Exception:
        print(f"Failed to process {entry.label}")


def build_preproc_pipeline():
    """
    Preprocesses and writes data offline.
    Iterates over all entries in the LJSpeech dataset.
    """
    p_root = get_project_root()
    _, f_out_dir = setup_dir_structure(p_root)

    entries = []
    with open(get_metadata_path(p_root), newline="") as f:
        md = csv.reader(f, delimiter="|")
        for row in md:
            # LJSpeech: label | text | normalized_text
            label, text = row[0], row[2] if len(row) > 2 else row[1]
            entries.append(MetadataEntry(label, text, get_wav_path(p_root, label)))

    results = []
    with Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(partial(preproc_entry, out_dir=f_out_dir), entries),
                total=len(entries),
            )
        )
    return f_out_dir


def load_dataset(batch_size=32, r=5):
    feat_dir = get_feat_dir(get_project_root())
    md_path = get_metadata_path(get_project_root())

    # Map each label to its text length for bucketing.
    text_lens = {}
    with open(md_path, newline="") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            label = row[0]
            text = row[2] if len(row) > 2 else row[1]
            text_lens[label] = min(len(text), MAX_SEQ_LENGTH)

    # Group .npz paths into buckets of width 16 chars.
    buckets = {}
    for p in feat_dir.glob("*.npz"):
        label = p.stem
        text_len = text_lens.get(label, MAX_SEQ_LENGTH)
        bucket_id = max(text_len - 1, 0) // 16
        buckets.setdefault(bucket_id, []).append(str(p))

    def load(path):
        with np.load(path.decode()) as d:
            text = d["text"]
            # Recover unpadded length from trailing all-zero rows.
            text_len = int(np.count_nonzero(np.any(text != 0, axis=1)))
            return (
                text[:text_len].astype(np.float32),
                d["mels"].astype(np.float32),
                d["spectra"].astype(np.float32),
            )

    def ensure_shapes(text, mels, spectra):
        text.set_shape([None, 256])
        mels.set_shape([None, 80])
        spectra.set_shape([None, 1025])
        return text, mels, spectra

    def pad_to_r(text, mels, spectra):
        frames = tf.shape(mels)[1]
        pad = (r - frames % r) % r
        if tf.greater(pad, 0):
            mels = tf.pad(mels, [[0, 0], [0, pad], [0, 0]])
            spectra = tf.pad(spectra, [[0, 0], [0, pad], [0, 0]])
        return text, mels, spectra

    def format_padded_batch(text, mels, spectra):
        return (
            {"text": text, "mels": mels},
            {"pred_mels": mels, "pred_spectra": spectra},
        )

    datasets = []
    weights = []
    for bucket_id in sorted(buckets.keys()):
        paths = buckets[bucket_id]
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.shuffle(buffer_size=len(paths))
        ds = ds.map(
            lambda p: tf.numpy_function(load, [p], [tf.float32, tf.float32, tf.float32]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.map(ensure_shapes, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.padded_batch(
            batch_size,
            padded_shapes=([None, 256], [None, 80], [None, 1025]),
            padding_values=(np.float32(0), np.float32(0), np.float32(0)),
        )
        ds = ds.map(pad_to_r, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(format_padded_batch, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        datasets.append(ds)
        weights.append(len(paths))

    if not datasets:
        return tf.data.Dataset.from_tensor_slices([])

    return tf.data.Dataset.sample_from_datasets(datasets, weights=weights)


# Data processing pipeline
# End result: Batches of similarly sized sequences that are all padded to the same length
# Samples are bucketed by text length
# Sample data all packed to length of max length in bucket
# Batches are taken from buckets
# Data can be preprocessed but needs to be labelled for sorting into buckets


def format_batch(text, mels, spectra):
    # Set shapes that tf.numpy_function stripped (per-example, before batching)
    text.set_shape([256, 256])
    mels.set_shape([800, 80])
    spectra.set_shape([800, 1025])

    return (
        {"text": text, "mels": mels},  # inputs (dict)
        {"pred_mels": mels, "pred_spectra": spectra},  # targets (keys match model output names)
    )


def build_dataset(batch_size=32):
    feat_dir = get_feat_dir(get_project_root())

    def load(p):
        d = np.load(p.decode())
        return d["text"], d["mels"], d["spectra"]

    ds = tf.data.Dataset.list_files(str(feat_dir / "*.npz"), shuffle=True)
    ds = ds.map(
        lambda p: tf.numpy_function(load, [p], [tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(format_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    build_preproc_pipeline()


if __name__ == "__main__":
    main()
