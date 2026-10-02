"""
Tacotron 1 layers and model.
This time around I am trying the vectorized approach instead of the looped approach.
"""

import keras
from keras import layers


class PreNet(layers.Layer):
    def __init__(self, input_dim=256, units=128):
        super().__init__()
        self.fc1 = layers.Dense(input_dim, activation="relu")
        self.d1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(units, activation="relu")
        self.d2 = layers.Dropout(0.5)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.d1(x, training=True)
        x = self.fc2(x)
        x = self.d2(x, training=True)
        return x


class HighwayNet(layers.Layer):
    def __init__(self, units=128):
        super().__init__()
        self.h = layers.Dense(units, activation="relu")
        self.t = layers.Dense(
            units,
            activation="sigmoid",
        )

    def call(self, inputs):
        h = self.h(inputs)
        t = self.t(inputs)
        return h * t + inputs * (1.0 - t)


class Cbhg(layers.Layer):
    def __init__(self, k=16, projection_units=(128, 128), units=128):
        super().__init__()
        self.conv_bank = [
            keras.Sequential(
                [
                    layers.Conv1D(
                        units,
                        i,
                        strides=1,
                        padding="same",
                        use_bias=False,
                    ),
                    layers.BatchNormalization(axis=-1),
                    layers.ReLU(),
                ]
            )
            for i in range(1, k + 1)
        ]
        self.pool = layers.MaxPool1D(pool_size=2, strides=1, padding="same")
        self.proj1 = keras.Sequential(
            [
                layers.Conv1D(
                    projection_units[0],
                    3,
                    strides=1,
                    padding="same",
                    use_bias=False,
                ),
                layers.BatchNormalization(axis=-1),
                layers.ReLU(),
            ]
        )
        self.proj2 = layers.Conv1D(projection_units[1], 3, padding="same", use_bias=True)
        self.residual_proj = layers.Dense(projection_units[1], use_bias=False)
        self.hnet = [HighwayNet(units) for _ in range(4)]
        self.gru = layers.Bidirectional(layers.GRU(units // 2, return_sequences=True))

    def call(self, inputs, training=False):
        x = [conv(inputs, training=training) for conv in self.conv_bank]
        x = layers.concatenate(x)

        x = self.pool(x)

        x = self.proj1(x, training=training)
        x = self.proj2(x)

        # Project inputs if channel dimensions don't match proj2 output
        residual = inputs
        if inputs.shape[-1] != x.shape[-1]:
            residual = self.residual_proj(inputs)

        x = x + residual

        for hw in self.hnet:
            x = hw(x)

        outputs = self.gru(x)
        return outputs


class Encoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.prenet = PreNet()
        self.cbhg = Cbhg(16, (128, 128), 128)

    def call(self, inputs, training=None):
        # inputs: (B, 256, 256)
        x = self.prenet(inputs, training=training)
        encoder_outputs = self.cbhg(x, training=training)
        return encoder_outputs


class DecoderCell(layers.Layer):
    # Decoder cell to be wrapped by RNN layer for sequence processing
    def __init__(self, units=256, n_mels=80, r=5):
        super().__init__()
        self.enc_out_proj = None  # Set per-batch
        self.attn_rnn = layers.GRUCell(units)
        self.dec_rnn1 = layers.GRUCell(units)
        self.dec_rnn2 = layers.GRUCell(units)
        self.attn = layers.AdditiveAttention()

        # Projection layers
        self.attn_proj = layers.Dense(units, activation="linear")
        self.dec_proj = layers.Dense(units, activation="linear")
        self.frame_proj = layers.Dense(r * n_mels, activation="linear")

        # Rnncell state spec: (s_attn_rnn, s_dec1_rnn, s_dec2_rnn)
        self.state_size = (units, units, units)
        self.output_size = r * n_mels

    # NOTE
    # As we concatenate the attn context and query vectors as input to the dec_rnn:
    # Better to project down for input to dec_rnn?
    # Or better to project attn_query down so that concat dim = U?
    def call(self, inputs, states):
        # inputs: mels (B, r * n_mels)
        # states: [s_attn_rnn, s_dec_rnn1, s_dec_rnn2]
        # constants: enc_keys_proj (B, T, U)

        s_a, s_d1, s_d2 = states

        # Generate attention query and attention context vector
        x = self.attn_proj(inputs)  # (B, U)
        q, [s_a_next] = self.attn_rnn(x, states=[s_a])  # (B, U)
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
    def __init__(self, r=5, n_mels=80, decoder_dim=256):
        super().__init__()
        self.decoder_dim = 256
        self.prenet = PreNet()
        self.dec_cell = DecoderCell(decoder_dim, n_mels, r)
        self.dec_rnn = layers.RNN(self.dec_cell, return_sequences=True, return_state=True)
        self.enc_key_proj = layers.Dense(decoder_dim, activation="linear")

    def call(self, target_mels, enc_outputs, initial_state=None):
        # target_mels: (B, T_dec, r * n_mels)
        # encoder_outputs: (B, T, D)

        self.dec_cell.enc_out_proj = self.enc_key_proj(enc_outputs)

        outputs, *states = self.dec_rnn(
            target_mels,
            initial_state=initial_state,
        )
        return outputs, states


class PostNet(layers.Layer):
    def __init__(self, num_freq=1025):
        super().__init__()
        self.input_proj = layers.Dense(128, use_bias=True)
        self.cbhg = Cbhg(8, (256, 128), 128)
        self.output_proj = layers.Dense(num_freq, use_bias=True)

    def call(self, inputs, training=True):
        # inputs: (B, T, r * 80)
        x = self.input_proj(inputs)
        x = self.cbhg(x, training=training)
        x = self.output_proj(x)
        return x


class TextToSpectra(keras.Model):
    def __init__(self, num_freq=1025):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.post_net = PostNet(num_freq)
        self.r = 5
        self.n_mels = 80

    def call(self, inputs, r=5, training=False):
        # inputs[0] (text): (B, T=256, D=256)
        # inputs[1] (mels): (B, n_frames=800, n_mels=80) teacher forcing and loss
        if training:
            text = inputs["text"]
            mels = inputs["mels"]
        else:
            text = inputs if not isinstance(inputs, dict) else inputs.get("text", inputs)
            return self.generate(text, r=r)

        n_frames = mels.shape[1]
        n_mels = mels.shape[2]

        ## Reshape mels and append <GO> frame
        # Reshape mels: predict `r` frames at each time step
        mels = keras.ops.reshape(
            mels, (-1, n_frames // r, r * n_mels)
        )  # (B, T_dec=n_frames // r, r * n_mels)
        # Drop last frame and prepend 0 frame
        mels = mels[:, :-1, :]  # type: ignore
        zero_frame = keras.ops.zeros((keras.ops.shape(mels)[0], 1, r * n_mels))  # type: ignore
        mels = keras.ops.concatenate([zero_frame, mels], axis=1)

        enc_seq = self.encoder(text, training=training)  # (B, T, 128)

        pred_mels, _ = self.decoder(mels, enc_seq, None)  # (B, T_dec, r * n_mels)

        # TODO Unsure what shape is best to feed the mels to the post net
        pred_mels = keras.ops.reshape(pred_mels, (-1, n_frames, n_mels))  # (B, n_frames, n_mels)
        pred_spectra = self.post_net(pred_mels)

        # Reshape outputs for loss calculation
        return {"pred_mels": pred_mels, "pred_spectra": pred_spectra}


# TODO For tomorrow
# Implement iterative inference generate() method.
