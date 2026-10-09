import torch
import torch.nn as nn

from taco.decoder import Attention
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
    batch_size = 5
    key_d = 256
    q_d = 1024
    attn = Attention(q_d, key_d)

    keys = torch.randn(batch_size, 100, key_d)
    q = torch.randn(batch_size, 1024)
    w_prev = torch.randn(batch_size, 100)

    c, w = attn(q, keys, w_prev)


def main():
    # model = TTSModel(DataConfig.n_vocab)
    # train(model)
    test_attn()


if __name__ == "__main__":
    main()
