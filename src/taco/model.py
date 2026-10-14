import torch
import torch.nn as nn

from taco.data import get_metadata_path, get_project_root, get_wav_root
from taco.dataset import TTSDataset, collate_fn
from taco.decoder import Decoder, PostNet
from taco.encoder import Encoder
from taco.hparams import DataConfig, ModelConfig
from taco.utils import get_logs_dir, plot_attention


class TTSModel(nn.Module):
    def __init__(self, vocab_size=128):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(
            ModelConfig.prenet_d, ModelConfig.attn_d, ModelConfig.lstm_d, DataConfig.n_mels
        )
        self.post_net = PostNet()

    def forward(self, text, text_lengths, target_mels=None):
        enc_seq = self.encoder(text, text_lengths)
        if target_mels is None:
            # Autoregressive inference is not implemented in the decoder yet.
            return enc_seq

        batch_size = target_mels.size(0)
        prev_frame = target_mels.new_zeros(batch_size, DataConfig.n_mels)
        pred_mels, pred_stops, attn_weights = self.decoder(prev_frame, enc_seq, target_mels)
        pred_mels_wres = self.post_net(pred_mels)

        return (
            pred_mels.transpose(1, 2),
            pred_mels_wres.transpose(1, 2),
            pred_stops.squeeze(1),
            attn_weights,
        )


def pick_short_indices(dataset, n):
    """Pick the n utterances with the shortest transcripts (cheap proxy for short audio)."""
    rows = dataset.labels_and_text
    lengths = [len(str(rows.iloc[i, 1]).strip()) for i in range(len(rows))]
    return sorted(range(len(rows)), key=lambda i: lengths[i])[:n]


def build_overfit_batch(n=2):
    project_root = get_project_root()
    dataset = TTSDataset(get_metadata_path(project_root), get_wav_root(project_root))
    indices = pick_short_indices(dataset, n)
    batch = [dataset[i] for i in indices]
    return collate_fn(batch)


def train(model, batch, device, steps=1500, lr=1e-3, log_every=50):
    texts, mels, text_lengths, mel_lengths = batch
    texts = texts.to(device)
    text_lengths = text_lengths.to(device)
    mels = mels.to(device)
    mel_lengths = mel_lengths.to(device)

    batch_size, n_frames, _ = mels.shape
    stop_targets = torch.zeros((batch_size, n_frames), device=device)
    for i in range(batch_size):
        stop_targets[i, mel_lengths[i] - 1] = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for step in range(steps):
        pred_mels, pred_mels_wres, pred_stops, attn_weights = model(texts, text_lengths, mels)

        loss_mel = mse(pred_mels, mels)
        loss_res_mel = mse(pred_mels_wres, mels)
        loss_stop = bce(pred_stops, stop_targets)
        loss = loss_mel + loss_stop + loss_res_mel

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            print(
                f"step {step:5d} | mel {loss_mel.item():.4f} | res_mel {loss_res_mel.item():.4f} "
                f"| stop {loss_stop.item():.4f} | total {loss.item():.4f}"
            )
            plot_attention(
                attn_weights,
                get_logs_dir() / f"attention_{step:05d}.png",
                title=f"Attention at step {step}",
            )
    return model


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch = build_overfit_batch(n=2)
    texts, mels, _, mel_lengths = batch
    print(
        f"Overfit batch: texts={tuple(texts.shape)}, "
        f"mels={tuple(mels.shape)}, mel_lengths={mel_lengths.tolist()}"
    )

    model = TTSModel(DataConfig.n_vocab).to(device)
    train(model, batch, device)


if __name__ == "__main__":
    main()
