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
        self.state_size = (units, units, units)
        self.output_size = r * n_mels

    def call(self, inputs, states):
        # inputs (mel features): (B, D=128)
        # states: [s_attn_rnn, s_dec_rnn1, s_dec_rnn2]
        # outputs: (B, r * n_mels), states

        s_a, s_d1, s_d2 = states

        # Generate attention query and attention context vector
        q, [s_a_next] = self.attn_rnn(inputs, states=[s_a])  # (B, U=256)
        q_3 = keras.ops.expand_dims(q, axis=1)  # (B, 1, U)
        c = self.attn([q_3, self.enc_out_proj], training=True)  # (B, 1, U)
        c = keras.ops.squeeze(c, axis=1)  # (B, U)

        # Generate output frames from decoder rnn
        dec_rnn_in = keras.ops.concatenate([c, q], axis=-1)  # (B, 2 * U)
        dec_rnn_in = self.dec_proj(dec_rnn_in)  # (B, U)
        x1, [s_d1_next] = self.dec_rnn1(dec_rnn_in, states=[s_d1])
        x1 = x1 + dec_rnn_in
        x2, [s_d2_next] = self.dec_rnn2(x1, states=[s_d2])
        x2 = x2 + x1
        frames = self.frame_proj(x2)  # (B, r * n_mels)

        return frames, (s_a_next, s_d1_next, s_d2_next)


class Decoder(layers.Layer):
    def __init__(self, r=5, n_mels=80, units=256):
        super().__init__()
        self.decoder_dim = 256
        self.prenet = PreNet()
        self.dec_cell = DecoderCell(r, n_mels, units)
        self.dec_rnn = layers.RNN(self.dec_cell, return_sequences=True, return_state=True)

    def call(self, target_mels, enc_outputs, initial_state=None):
        # target_mels: (B, T_dec, n_mels)
        # encoder_outputs: (B, T, D)

        # NOTE
        # Shouldn't we have a decoder cell per batch sample?
        # Each sample has its own encoded sequence that we want to decode?
        # Maybe this is taken care of?

        x = self.prenet(target_mels)
        self.dec_cell.enc_out_proj = enc_outputs
        outputs, *states = self.dec_rnn(x, initial_state=initial_state)  # (B, T_dec, r * n_mels)

        return outputs, states


class PostNet(layers.Layer):
    def __init__(self, r=5, n_mels=80, num_freq=1025):
        super().__init__()
        self.r = r
        self.n_mels = n_mels
        self.input_proj = layers.Dense(128, activation="linear")
        self.cbhg = Cbhg(8, (256, 128), 128)
        self.output_proj = layers.Dense(num_freq, activation="linear")

    def call(self, inputs, training=True):
        # inputs: (B, T_dec, r * n_mels)
        # outputs: (B, T, n_fft)

        shape = keras.ops.shape(inputs)
        inputs = keras.ops.reshape(inputs, (shape[0], shape[1] * self.r, self.n_mels))
        x = self.input_proj(inputs)
        x = self.cbhg(x, training=training)
        x = self.output_proj(x)
        return x
