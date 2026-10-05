import keras
from keras import layers


class PreNet(layers.Layer):
    def __init__(self):
        super().__init__()
        self.fc1 = layers.Dense(256, activation="relu")
        self.d1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(128, activation="relu")
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
        self.gru = layers.Bidirectional(layers.GRU(units, return_sequences=True))

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

        outputs = self.gru(x)  # (B, T, 2 * U)
        return outputs
