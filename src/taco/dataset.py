# LJSpeech dataset
import random
from pathlib import Path

import librosa
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from taco.data import get_metadata_path, get_project_root, get_wav_root
from taco.hparams import DataConfig


class TTSDataset(Dataset):
    def __init__(self, md_file: Path, audio_dir: Path):
        self.labels_and_text = pd.read_csv(md_file, sep="|")
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.labels_and_text)

    def __getitem__(self, idx):
        row = self.labels_and_text.iloc[idx]
        text = row[2] if row[2] else row[1]

        audio_file = Path(row[0]).with_suffix(".wav")
        audio_path = self.audio_dir / audio_file
        audio = librosa.load(audio_path, sr=DataConfig.sr)
        return text, audio


class RandomBucketSampler(Sampler):
    def __init__(self, data, batch_size, key=lambda x: len(x[0])):  # Key on text len
        self.data = data
        self.batch_size = batch_size
        self.key = key

    def __iter__(self):
        indices = list(range(len(self.data)))
        random.shuffle(indices)

        pool_size = self.batch_size * 100
        for i in range(0, len(indices), pool_size):
            pool = indices[i : i + pool_size]
            pool.sort(key=lambda idx: self.key(self.data[idx]))

            batches = [pool[j : j + self.batch_size] for j in range(0, len(pool), self.batch_size)]
            random.shuffle(batches)

            yield from batches

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


def collate_fn(batch):
    texts, audios = zip(*batch, strict=True)

    text_lengths = torch.tensor([len(x) for x in texts])
    audio_lengths = torch.tensor([len(x) for x in audios])

    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    audios = pad_sequence(audios, batch_first=True)

    return texts, audios, text_lengths, audio_lengths


def make_data_loader():
    project_root = get_project_root()
    dataset = TTSDataset(get_metadata_path(project_root), get_wav_root(project_root))

    sampler = RandomBucketSampler(dataset, batch_size=32, key=lambda x: len(x[1]))

    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

    return loader


def main():
    loader = make_data_loader()


if __name__ == "__main__":
    main()
