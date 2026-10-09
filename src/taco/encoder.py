import torch.nn as nn

from taco.hparams import ModelConfig


class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        units = 512
        embed_dim = ModelConfig.embedding_d

        self.embeddings = nn.Embedding(vocab_size, embed_dim, 0)
        self.convs = nn.Sequential(
            nn.Conv1d(embed_dim, units, kernel_size=5, padding=2),
            nn.BatchNorm1d(units),
            nn.ReLU(),
            nn.Conv1d(units, units, kernel_size=5, padding=2),
            nn.BatchNorm1d(units),
            nn.ReLU(),
            nn.Conv1d(units, units, kernel_size=5, padding=2),
            nn.BatchNorm1d(units),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(units, units // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        # x: (B, Tt, 128)
        embeddings = self.embeddings(x)
        outputs = self.convs(embeddings)
        outputs, _ = self.lstm(outputs)
        return outputs
