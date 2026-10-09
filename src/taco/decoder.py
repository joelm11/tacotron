import torch
import torch.nn as nn

from taco.hparams import ModelConfig


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, attn_dim=128):
        super().__init__()
        location_feat_d = ModelConfig.loc_feat_d
        filter_sz = ModelConfig.decoder_conv_filter_sz
        self.w_q = nn.Linear(query_dim, attn_dim, bias=False)
        self.w_k = nn.Linear(key_dim, attn_dim)
        self.w_l = nn.Linear(location_feat_d, attn_dim)
        self.w_v = nn.Linear(attn_dim, 1)
        self.loc_conv = nn.Conv1d(attn_dim, location_feat_d, kernel_size=filter_sz, padding=15)

    def forward(self, q, k, prev_weights):
        # q: (B, q_dim)
        # k: (B, Tt, k_dim)
        # p_w: (B, Tt, 1)
        # NOTE Likely need to do something with dims here?
        loc_feats = self.loc_conv(prev_weights)

        q = self.w_q(q).unsqueeze(1)  # (B, 1, U)
        k = self.w_k(k)  # (B, Tt, U)
        loc_feats = self.w_l(loc_feats).unsqueeze(1)  # (B, 1, U)

        score = self.w_v(torch.tanh(q + k + loc_feats))  # (B, Tt, 1)
        weights = torch.softmax(score, dim=-1)
        c = torch.sum(weights * k)
        return c, weights


class Decoder(nn.Module):
    def __init__(self, prenet_dim, attn_dim, lstm_dim, n_mels):
        super().__init__()
        self.prenet = nn.Sequential(
            nn.Linear(n_mels, prenet_dim),
            nn.ReLU(),
            nn.Linear(prenet_dim, prenet_dim),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(prenet_dim + attn_dim, lstm_dim, num_layers=2, batch_first=True)

        self.proj_mels = nn.Linear(lstm_dim + attn_dim, n_mels)
        self.proj_stop = nn.Linear(lstm_dim + attn_dim, 1)

    def forward(self, x):
        return
