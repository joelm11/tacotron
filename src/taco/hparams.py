from dataclasses import dataclass


@dataclass
class DataConfig:
    # Audio
    sr = 22050
    frame_ms = 50
    frame_hop_ms = 12.5
    n_mels = 80
    # Text
    max_text_len = 128


@dataclass
class ModelConfig:
    embedding_d = 512
