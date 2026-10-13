import torch
import torch.nn as nn

from taco.decoder import Decoder
from taco.encoder import Encoder
from taco.hparams import DataConfig, ModelConfig


class TTSModel(nn.Module):
    def __init__(self, vocab_size=128):
        super().__init__()
        self.encoder = Encoder(vocab_size)

    def forward(self, x):
        x = self.encoder(x)
        return x


def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    for step in range(10):
        # Fake input and target batch
        x = torch.randint(0, DataConfig.n_vocab, (1, DataConfig.max_text_len))
        y = torch.randn(1, DataConfig.max_text_len, ModelConfig.embedding_d)

        # Forward
        encoded_seq = model(x)

        loss = criterion(encoded_seq, y)

        # Backward + parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step}: loss={loss.item():.4f}")

    return


def test_attn():
    batch_size, pn_dim, attn_dim, lstm_dim, n_mels = [16, 256, 128, 1024, 80]

    dec = Decoder(pn_dim, attn_dim, lstm_dim, n_mels)
    enc_seq = torch.randn(batch_size, 100, 512)
    target_seq = torch.randn(batch_size, 200, n_mels)
    prev_frame = torch.zeros((batch_size, n_mels))

    dec.forward(prev_frame, enc_seq, target_seq)


def main():
    # model = TTSModel(DataConfig.n_vocab)
    # train(model)
    test_attn()


if __name__ == "__main__":
    main()
