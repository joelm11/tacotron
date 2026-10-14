import torch
import torch.nn as nn

from taco.encoder import Encoder
from taco.hparams import DataConfig, ModelConfig


# TODO Add cumulative attention weights
# TODO Add masking
class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, attn_dim=128):
        super().__init__()

        self.units = attn_dim
        location_feat_d = ModelConfig.loc_feat_d
        filter_sz = ModelConfig.decoder_conv_filter_sz
        self.w_q = nn.Linear(query_dim, attn_dim, bias=False)
        self.w_k = nn.Linear(key_dim, attn_dim, bias=False)
        self.w_l = nn.Linear(location_feat_d, attn_dim, bias=False)
        self.w_v = nn.Linear(attn_dim, 1, bias=False)
        self.loc_conv = nn.Conv1d(1, location_feat_d, kernel_size=filter_sz, padding=15)

    def forward(self, q, k, prev_weights):
        # q: (B, q_dim)
        # k: (B, Tt, k_dim)
        # p_w: (B, Tt, 1)

        loc_feats = self.loc_conv(prev_weights.transpose(1, 2))

        q = self.w_q(q).unsqueeze(1)  # (B, 1, U)
        k = self.w_k(k)  # (B, Tt, U)
        loc_feats = self.w_l(loc_feats.transpose(1, 2))  # (B, 1, U)

        score = self.w_v(torch.tanh(q + k + loc_feats))  # (B, Tt, 1)
        weights = torch.softmax(score, dim=1)

        c = torch.einsum("bsd,bsq->bd", k, weights)

        return c, weights


class Decoder(nn.Module):
    def __init__(self, prenet_dim, attn_dim, lstm_dim, n_mels):
        super().__init__()

        self.lstm_dim = lstm_dim
        self.attn_dim = attn_dim
        self.prenet = nn.Sequential(
            nn.Linear(n_mels, prenet_dim),
            nn.ReLU(),
            nn.Linear(prenet_dim, prenet_dim),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(prenet_dim + attn_dim, lstm_dim, num_layers=2, batch_first=True)
        self.attn = Attention(query_dim=lstm_dim, key_dim=Encoder.OUT_D)
        self.proj_mels = nn.Linear(lstm_dim + attn_dim, n_mels)
        self.proj_stop = nn.Linear(lstm_dim + attn_dim, 1)

    def forward(self, prev_frame, enc_seq, target_mels=None):
        if target_mels is None:
            return  # Inference TBI

        B, S, E = enc_seq.shape
        B, T, _ = target_mels.shape

        device = enc_seq.device
        ctx_prev = torch.zeros((B, self.attn_dim), device=device)
        h_prev = torch.zeros((2, self.lstm_dim), device=device)
        c_prev = torch.zeros((2, self.lstm_dim), device=device)
        w_prev = torch.zeros((B, S, 1), device=device)

        pred_frames = []
        pred_stops = []
        attn_weights = []
        for t in range(T):
            x = self.prenet(prev_frame)
            x = torch.cat((x, ctx_prev), -1)

            q, (h_prev, c_prev) = self.rnn(x, (h_prev, c_prev))
            c, w = self.attn(q, enc_seq, w_prev)
            w_prev = w_prev + w
            qc = torch.cat((q, c), -1)

            pred_frame = self.proj_mels(qc)
            pred_stop = self.proj_stop(qc)

            pred_frames.append(pred_frame)
            pred_stops.append(pred_stop)
            attn_weights.append(w)

            if target_mels is not None:
                prev_frame = target_mels[:, t]
            else:
                prev_frame = pred_frame

        # attn_weights: (B, T, S) alignment maps accumulated over the decoder steps.
        return (
            torch.stack(pred_frames, dim=-1),
            torch.stack(pred_stops, dim=-1),
            torch.stack(attn_weights, dim=1).squeeze(-1),
        )


class PostNet(nn.Module):
    def __init__(self):
        super().__init__()

        n_mels = DataConfig.n_mels
        channels = ModelConfig.postnet_conv_n_ch
        k_sz = ModelConfig.postnet_conv_filter_sz

        self.convs = nn.Sequential(
            nn.Conv1d(n_mels, channels, k_sz, padding="same"),
            nn.BatchNorm1d(channels),
            nn.Tanh(),
            nn.Conv1d(channels, channels, k_sz, padding="same"),
            nn.BatchNorm1d(channels),
            nn.Tanh(),
            nn.Conv1d(channels, channels, k_sz, padding="same"),
            nn.BatchNorm1d(channels),
            nn.Tanh(),
            nn.Conv1d(channels, channels, k_sz, padding="same"),
            nn.BatchNorm1d(channels),
            nn.Tanh(),
            nn.Conv1d(channels, channels, k_sz, padding="same"),
            nn.BatchNorm1d(channels),
        )
        self.proj = nn.Linear(channels, n_mels, bias=False)

    def forward(self, pred_mels):
        # pred_mels: (B, N, T)

        residuals = self.convs(pred_mels)
        residuals = self.proj(residuals.transpose(1, 2))
        return pred_mels + residuals.transpose(1, 2)
