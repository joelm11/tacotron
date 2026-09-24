"""Tacotron layers and model."""

import keras
from keras import layers

### Model layers


class PreNet(layers.Layer):
    def __init__(self, units=128):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.fc1 = layers.Dense(input_shape[-1], activation="relu")
        self.d1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(self.units, activation="relu")
        self.d2 = layers.Dropout(0.5)

    # To the encoder/decoder prenets we expect input to be of shape (B, T, 256)
    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.d1(x, training)
        x = self.fc2(x)
        x = self.d2(training)
        return x


class HighwayNet(layers.Layer):
    def __init__(self, units=128):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.h = layers.Dense(self.units, activation="relu")
        init_bias = keras.initializers.Constant(value=-2.0)
        self.t = layers.Dense(
            self.units,
            activation="sigmoid",
            bias_initializer=init_bias,  # type: ignore
        )

    def call(self, inputs, training=None):
        h = self.h(inputs)
        t = self.t(inputs)
        return h * t + inputs * (1.0 - t)


class Cbhg(layers.Layer):
    """*Encoder* CBHG. Input to this layer is (B, T, 128). Output from this layer is (B, T, 128)"""

    def __init__(self, k=16, units=128):
        super().__init__()
        self.units = units
        self.k = k
        self.conv_bank = [
            keras.Sequential(
                [
                    layers.Conv1DTranspose(
                        self.units,
                        i,
                        strides=1,
                        padding="same",
                        use_bias=False,
                    ),
                    layers.BatchNormalization(axis=-1),
                    layers.ReLU(),
                ]
            )
            for i in range(1, self.k + 1)
        ]
        self.pool = layers.MaxPool1D(pool_size=2, strides=1, padding="same")
        self.proj1 = keras.Sequential(
            [
                layers.Conv1D(
                    self.units,
                    3,
                    strides=1,
                    padding="same",
                    use_bias=False,
                ),
                layers.BatchNormalization(axis=-1),
                layers.ReLU(),
            ]
        )
        self.proj2 = layers.Conv1D(self.units, 3, padding="same", use_bias=True)
        self.hnet = [HighwayNet(self.units) for _ in range(4)]
        self.gru = layers.Bidirectional(layers.GRU(self.units // 2, return_sequences=True))

    def call(self, inputs, training=None):
        # After conv bank (B, T, K, 128)
        # Concatenate along K so we have (B, T, 2048)
        x = [conv(inputs) for conv in self.conv_bank]
        x = layers.concatenate(x)

        x = self.pool(x)

        x = self.proj1(x)
        x = self.proj2(x)

        x = x + inputs

        for hw in self.hnet:
            x = hw(x)

        outputs = self.gru(x)
        return outputs


class DecoderRnn(layers.Layer):
    """2-layer residual GRU"""

    def __init__(self, cells=256):
        super().__init__()
        self.cells = cells

        self.g1 = layers.GRU(self.cells, return_sequences=True)
        self.g2 = layers.GRU(self.cells, return_sequences=True)

    def call(self, inputs):
        x1 = self.g1(inputs) + inputs
        x2 = self.g2(x1) + x1
        return x2


class AttentionRnn(layers.Layer):
    """Attention RNN"""

    def __init__(self, cells=256):
        super().__init__()
        self.cells = cells

        self.g1 = layers.GRU(self.cells, return_state=True)

    def call(self, inputs):
        _, s = self.g1(inputs)
        return s


class Attention(layers.Layer):
    def __init__(self, cells=256):
        super().__init__()
        self.attn = layers.AdditiveAttention()

    def call(self, query, keys):
        pass


class Encoder(layers.Layer):
    """Takes as input ohe text, encodes to a learned intermediate representation."""

    def __init__(self):
        pass

    def call(self, inputs):
        pass


class Decoder(layers.Layer):
    """Takes as input encoder representation of sequence, other data for calculating loss."""

    def __init__(self):
        pass

    def call(self, inputs):
        pass


# class Tacotron(keras.Model):
