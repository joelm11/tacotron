# The decoder is implemented implemented as an rnn cell.
# It is then wrapped with an rnn layer to operate over the input sequence.


import keras
from keras import layers

from ops import Cbhg, PreNet


class DecoderCell(layers.Layer):
    # Decoder cell to be wrapped by RNN layer for sequence processing
    def __init__(self, r=5, n_mels=80, units=256):
        super().__init__()
        # Set per-batch
        self.enc_out_proj = None  # (B, T_enc, D=256)

        self.attn_rnn = layers.GRUCell(units)
        self.dec_rnn1 = layers.GRUCell(units)
        self.dec_rnn2 = layers.GRUCell(units)
        self.attn = layers.AdditiveAttention()

        # Projection layers
        self.dec_proj = layers.Dense(units, activation="linear")
        self.frame_proj = layers.Dense(r * n_mels, activation="linear")

        # Declare cell specs
        self.state_size = (units, units, units, units)
        self.output_size = r * n_mels

    def call(self, inputs, states, training=None):
        # inputs (mel features): (B, D=128)
        # states: [s_attn_rnn, s_dec_rnn1, s_dec_rnn2, c_prev]
        s_a, s_d1, s_d2, c_prev = states

        # Attention RNN: pre-net output concatenated with PREVIOUS context
        attn_input = keras.ops.concatenate([inputs, c_prev], axis=-1)  # (B, 128+256)
        q, [s_a_next] = self.attn_rnn(attn_input, states=[s_a])  # (B, U)

        # Compute attention from NEW attention-RNN state (Bahdanau style)
        q_3 = keras.ops.expand_dims(q, axis=1)  # (B, 1, U)
        c = self.attn([q_3, self.enc_out_proj], training=training)  # (B, 1, U)
        c = keras.ops.squeeze(c, axis=1)  # (B, U)

        # Decoder RNNs — input is projection of [context, attn-rnn output]
        # This mirrors AttentionWrapper output of the reference implementation
        dec_rnn_in = keras.ops.concatenate([c, q], axis=-1)  # (B, 2*U)
        dec_rnn_in = self.dec_proj(dec_rnn_in)  # (B, U)

        x1, [s_d1_next] = self.dec_rnn1(dec_rnn_in, states=[s_d1])
        x1 = x1 + dec_rnn_in
        x2, [s_d2_next] = self.dec_rnn2(x1, states=[s_d2])
        x2 = x2 + x1
        frames = self.frame_proj(x2)  # (B, r * n_mels)

        return frames, (s_a_next, s_d1_next, s_d2_next, c)


class Decoder(layers.Layer):
    def __init__(self, r=5, n_mels=80, units=256):
        super().__init__()
        self.prenet = PreNet()
        self.dec_cell = DecoderCell(r, n_mels, units)
        self.dec_rnn = layers.RNN(self.dec_cell, return_sequences=True, return_state=True)

    def call(self, target_mels, enc_outputs, initial_state=None, training=None):
        # target_mels: (B, T_dec, n_mels)
        # encoder_outputs: (B, T, D)

        x = self.prenet(target_mels, training=training)
        self.dec_cell.enc_out_proj = enc_outputs
        outputs, *states = self.dec_rnn(x, initial_state=initial_state, training=training)

        return outputs, states


class PostNet(layers.Layer):
    def __init__(self, r=5, n_mels=80, num_freq=1025):
        super().__init__()
        self.r = r
        self.n_mels = n_mels
        # Post-net CBHG: no residual, 2nd projection to n_mels, extra dense to 128
        self.cbhg = Cbhg(8, (128, n_mels), 128, use_residual=False, extra_dense_units=128)
        self.output_proj = layers.Dense(num_freq, activation="linear")

    def call(self, inputs, training=None):
        # inputs: (B, T_dec, r * n_mels)
        shape = keras.ops.shape(inputs)
        inputs = keras.ops.reshape(inputs, (shape[0], shape[1] * self.r, self.n_mels))
        x = self.cbhg(inputs, training=training)
        x = self.output_proj(x)
        return x
