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

    def forward(self, x):
        embeddings = self.embeddings(x)
        embeddings = embeddings.permute(0, 2, 1)
        outputs = self.convs(embeddings)
        outputs = outputs.permute(0, 2, 1)
        outputs, _ = self.lstm(outputs)
        return outputs
