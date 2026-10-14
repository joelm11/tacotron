import torch
import torch.nn as nn

from taco.hparams import ModelConfig


class Encoder(nn.Module):
    UNITS = ModelConfig.embedding_d
    OUT_D = UNITS

    def __init__(self, vocab_size):
        super().__init__()
        units = self.UNITS
        embed_dim = ModelConfig.embedding_d
        filter_sz = ModelConfig.encoder_conv_filter_sz

        self.embeddings = nn.Embedding(vocab_size, embed_dim, 0)
        self.convs = nn.Sequential(
            nn.Conv1d(embed_dim, units, kernel_size=filter_sz, padding=2),
            nn.BatchNorm1d(units),
            nn.ReLU(),
            nn.Conv1d(units, units, kernel_size=filter_sz, padding=2),
            nn.BatchNorm1d(units),
            nn.ReLU(),
            nn.Conv1d(units, units, kernel_size=filter_sz, padding=2),
            nn.BatchNorm1d(units),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(units, units // 2, bidirectional=True, batch_first=True)

    def forward(self, text, text_lengths):
        embeddings = self.embeddings(text)
        embeddings = embeddings.permute(0, 2, 1)
        encoded_seq = self.convs(embeddings)
        encoded_seq = encoded_seq.permute(0, 2, 1)
        encoded_seq, _ = self.lstm(encoded_seq)
        seq_masks = torch.tensor((text.shape[0],))
        return encoded_seq, seq_masks
