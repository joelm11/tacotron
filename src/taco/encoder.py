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

    @staticmethod
    def make_seq_mask(text, text_lengths):
        """Boolean key mask for the padded sequence.

        Returns a (B, S, 1) tensor that is True at padded positions so it can be
        passed directly to ``Tensor.masked_fill`` in the attention module.
        """
        max_len = text.shape[1]
        # (B, S) -> True where position >= original length
        positions = torch.arange(max_len, device=text.device).unsqueeze(0)
        seq_masks = positions >= text_lengths.unsqueeze(1)
        return seq_masks.unsqueeze(-1)

    def forward(self, text, text_lengths):
        embeddings = self.embeddings(text)
        embeddings = embeddings.permute(0, 2, 1)
        encoded_seq = self.convs(embeddings)
        encoded_seq = encoded_seq.permute(0, 2, 1)
        encoded_seq, _ = self.lstm(encoded_seq)

        seq_masks = self.make_seq_mask(text, text_lengths)
        return encoded_seq, seq_masks
