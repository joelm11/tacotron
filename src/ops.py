import keras
from keras import layers


class PreNet(layers.Layer):
    def __init__(self):
        super().__init__()
        self.fc1 = layers.Dense(256, activation="relu")
        self.d1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(128, activation="relu")
        self.d2 = layers.Dropout(0.5)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.d1(x, training=training)
        x = self.fc2(x)
        x = self.d2(x, training=training)
        return x


class HighwayNet(layers.Layer):
    def __init__(self, units=128):
        super().__init__()
        self.h = layers.Dense(units, activation="relu")
        initz = keras.initializers.Constant(value=-1.0)
        self.t = layers.Dense(
            units,
            bias_initializer=initz,
            activation="sigmoid",
        )

    def call(self, inputs):
        h = self.h(inputs)
        t = self.t(inputs)
        return h * t + inputs * (1.0 - t)


class Cbhg(layers.Layer):
    def __init__(self, k=16, projection_units=(128, 128), units=128, use_residual=True, extra_dense_units=None):
        super().__init__()
        self.use_residual = use_residual
        self.extra_dense_units = extra_dense_units

        # Conv1D banks: no norm/ReLU per conv — that happens after concat
        self.conv_bank = [
            layers.Conv1D(
                units,
                i,
                strides=1,
                padding="same",
                use_bias=False,
            )
            for i in range(1, k + 1)
        ]
        self.norm_relu = keras.Sequential([
            layers.LayerNormalization(),
            layers.ReLU(),
        ])

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
                layers.LayerNormalization(),
                layers.ReLU(),
            ]
        )

        self.proj2 = layers.Conv1D(projection_units[1], 3, padding="same", use_bias=False)
        self.proj2_norm = layers.LayerNormalization()

        self.residual_proj = layers.Dense(projection_units[1], use_bias=False)

        if extra_dense_units is not None:
            self.extra_dense = layers.Dense(extra_dense_units, use_bias=True)
        else:
            self.extra_dense = None

        self.hnet = [HighwayNet(units) for _ in range(4)]
        self.gru = layers.Bidirectional(layers.GRU(units, return_sequences=True))

    def call(self, inputs, training=False):
        # Conv1D banks
        x = [conv(inputs, training=training) for conv in self.conv_bank]
        x = layers.concatenate(x)
        x = self.norm_relu(x, training=training)

        x = self.pool(x)

        x = self.proj1(x, training=training)
        x = self.proj2(x)
        x = self.proj2_norm(x, training=training)

        # Residual connection (matches encoder; disabled for post-net)
        if self.use_residual:
            residual = inputs
            if inputs.shape[-1] != x.shape[-1]:
                residual = self.residual_proj(inputs)
            x = x + residual

        # Extra affine transform for dimensionality sync (post-net only)
        if self.extra_dense is not None:
            x = self.extra_dense(x)

        for hw in self.hnet:
            x = hw(x)

        outputs = self.gru(x)
        return outputs
