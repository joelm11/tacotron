# LJSpeech dataset
from pathlib import Path

import librosa
import pandas as pd
from torch.utils.data import Dataset

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
