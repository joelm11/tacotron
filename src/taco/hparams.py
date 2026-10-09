from dataclasses import dataclass


@dataclass
class DataConfig:
    # Audio
    sr = 22050
    frame_ms = 50
    frame_hop_ms = 12.5
    n_mels = 80
    n_fft = 2048
    f_min = 125.0
    f_max = 7600
    # Text
    max_text_len = 128
    n_vocab = 128


@dataclass
class ModelConfig:
    embedding_d = 512


@dataclass
class TrainingConfig:
    batch_size = 32
