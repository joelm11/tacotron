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

    def call(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.d1(x, training=training)
        x = self.fc2(x)
        x = self.d2(x, training=training)
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
        self.hnet = [HighwayNet(units) for _ in range(4)]
        self.gru = layers.Bidirectional(layers.GRU(units // 2, return_sequences=True))

    def call(self, inputs, training=False):
        x = [conv(inputs, training=training) for conv in self.conv_bank]
        x = layers.concatenate(x)

        x = self.pool(x)

        x = self.proj1(x, training=training)
        x = self.proj2(x)

        x = x + inputs

        for hw in self.hnet:
            x = hw(x)

        outputs = self.gru(x)
        return outputs


class AttentionRnn(layers.Layer):
    def __init__(self, cells=256):
        super().__init__()
        self.cells = cells
        self.g1 = layers.GRU(self.cells, return_sequences=True, return_state=True)

    def call(self, inputs):
        # inputs: (B, T, D)
        x, s = self.g1(inputs)
        return x, s


class DecoderRnn(layers.Layer):
    def __init__(self, cells=256, num_mels=80, r=5):
        super().__init__()
        self.input_proj = layers.Dense(cells, activation="linear")
        self.g1 = layers.GRU(cells, return_sequences=True, return_state=True)
        self.g2 = layers.GRU(cells, return_sequences=True, return_state=True)
        self.frame_proj = layers.Dense(num_mels * r, activation=None)
        self.reshape = layers.Reshape((r, num_mels))

    def call(self, inputs):
        # inputs: (B, T, C + Q)
        # outputs: (B, T, r, num_mels)
        x = self.input_proj(inputs)
        x1, s1 = self.g1(x)
        x1 = x1 + x
        x2, s2 = self.g2(x1)
        x2 = x2 + x1
        outputs = self.frame_proj(x2)
        outputs = self.reshape(outputs)
        return outputs, (s1, s2)


class Attention(layers.Layer):
    # Seq2Seq attention mechanism
    def __init__(self):
        super().__init__()
        self.attn = layers.AdditiveAttention()

    def call(self, query, keys):
        # query: (B, T, D)
        # keys: (B, T, D)
        x = self.attn([query, keys])
        return x


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


class Decoder(layers.Layer):
    # TODO generate() method for inference
    def __init__(self):
        super().__init__()
        self.proj_enc = layers.Dense(256, activation="linear")
        self.proj1 = layers.Dense(256, activation="linear")
        self.prenet = PreNet()
        self.proj2 = layers.Dense(256, activation="linear")
        self.attn_rnn = AttentionRnn()
        self.attn = Attention()
        self.dec_rnn = DecoderRnn()

    def call(self, inputs, encoder_outputs, training=False):
        # inputs: (B, T, n_mels)
        # encoder_outputs: (B, T, 128)

        # Project target mels and encoder outputs
        x = self.proj1(inputs)
        encoder_outputs = self.proj_enc(encoder_outputs)

        x = self.prenet(x, training=training)  # (B, T, 128)
        x = self.proj2(x)  # (B, T, 256)
        q, s_a = self.attn_rnn(x)  # (B, T, 256)
        c = self.attn(q, encoder_outputs)  # (B, T, 256)
        x, s_d = self.dec_rnn(keras.ops.concatenate([c, q], axis=-1))  # (B, T, r, num_mels)
        return x, (s_a, s_d)

    def generate(self, inputs):
        pass

    def get_initial_states(self, batch_size):
        attn = keras.ops.zeros((batch_size, self.attn_rnn.cells))
        d1 = keras.ops.zeros((batch_size, self.dec_rnn.g1.units))
        d2 = keras.ops.zeros((batch_size, self.dec_rnn.g2.units))
        return (attn, (d1, d2))


class PostNet(layers.Layer):
    def __init__(self, num_freq=1025):
        super().__init__()
        self.input_proj = layers.Dense(128, use_bias=True)
        self.cbhg = Cbhg(8, (256, 128), 128)
        self.output_proj = layers.Dense(num_freq, use_bias=True)

    def call(self, inputs, training=False):
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

    def call(self, inputs, r=5, training=False):
        # inputs[0] (text): (B, T=256, D=256)
        # inputs[1] (mels): (B, n_frames=800, n_mels=80) teacher forcing and loss
        if training:
            text = inputs["text"]
            mels = inputs["mels"]
        else:
            # TODO generate() method for inference
            return

        batch_size = text.shape[0]
        t_dec = mels.shape[1]
        n_mels = mels.shape[2]

        # Reshape mels - predict `r` frames at each time step
        mels = keras.ops.reshape(mels, (batch_size, -1, r * n_mels))

        enc_seq = self.encoder(text)  # (B, T, 128)

        pred_mels, _ = self.decoder(mels, enc_seq, training)  # (B, n_frames // r, r * n_mels)
        # Reshape output to match target
        pred_mels = keras.ops.reshape(pred_mels, (batch_size, t_dec, -1))  # (B, n_frames, n_mels)

        # TODO Unsure what shape is best to feed the mels to the post net
        pred_spectra = self.post_net(pred_mels)

        return {"pred_mels": pred_mels, "pred_spectra": pred_spectra}

    def generate(self, inputs):
        pass
