"""Tacotron layers and model."""

import keras
import tensorflow as tf
from keras import layers

### Model layers


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
        init_bias = keras.initializers.Constant(value=-2.0)
        self.t = layers.Dense(
            units,
            activation="sigmoid",
            bias_initializer=init_bias,  # type: ignore
        )

    def call(self, inputs):
        h = self.h(inputs)
        t = self.t(inputs)
        return h * t + inputs * (1.0 - t)


class Cbhg(layers.Layer):
    """CBHG. Input to this layer is (B, T, 128). Output from this layer is (B, T, 128)"""

    def __init__(self, k=16, projection_units=(128, 128), units=128):
        super().__init__()
        self.units = units
        self.conv_bank = [
            keras.Sequential(
                [
                    layers.Conv1D(
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
        self.hnet = [HighwayNet(self.units) for _ in range(4)]
        self.gru = layers.Bidirectional(layers.GRU(self.units // 2, return_sequences=True))

    def call(self, inputs, training=False):
        # After conv bank (B, T, K, 128)
        # Concatenate along K so we have (B, T, 2048)
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


class DecoderRnn(layers.Layer):
    # 2 layer residual GRU
    def __init__(self, cells=256, num_mels=80, r=3):
        super().__init__()
        self.r = r
        self.input_proj = layers.Dense(cells, activation=None)
        self.g1 = layers.GRU(cells, return_sequences=True, return_state=True)
        self.g2 = layers.GRU(cells, return_sequences=True, return_state=True)
        self.frame_proj = layers.Dense(num_mels * r, activation=None)
        self.reshape = layers.Reshape((r, num_mels))

    def call(self, inputs, state):
        # inputs: (B, 1, features)
        x = self.input_proj(inputs)
        x1, s1 = self.g1(x, initial_state=state[0])
        x1 = x1 + x
        x2, s2 = self.g2(x1, initial_state=state[1])
        x2 = x2 + x1
        outputs = self.frame_proj(x2)
        outputs = self.reshape(outputs)
        return outputs, (s1, s2)


class AttentionRnn(layers.Layer):
    # Attention RNN
    def __init__(self, cells=256):
        super().__init__()
        self.cells = cells
        self.g1 = layers.GRU(self.cells, return_sequences=True, return_state=True)

    def call(self, inputs, state):
        # inputs: (B, 1, features)
        x, s = self.g1(inputs, initial_state=state)
        return x, s


class Attention(layers.Layer):
    # Seq2Seq attention mechanism
    def __init__(self):
        super().__init__()
        self.query_proj = layers.Dense(128, use_bias=False)
        self.attn = layers.AdditiveAttention()

    def call(self, query, keys):
        # AdditiveAttention expects 3-D inputs: [batch, seq_len, dim]
        query = self.query_proj(query)  # project to key/value dim
        query = keras.ops.expand_dims(query, axis=1)  # (B, 1, 128)
        x = self.attn([query, keys])  # (B, 1, 128)
        x = x[:, 0, :]  # (B, 128)
        return x


class Encoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.prenet = PreNet()
        self.cbhg = Cbhg(16, (128, 128), 128)

    # Inputs: (B, 256, 256)
    def call(self, inputs, training=None):
        x = self.prenet(inputs, training=training)
        encoder_outputs = self.cbhg(x, training=training)
        return encoder_outputs


class Decoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.prenet = PreNet()
        self.attn_rnn = AttentionRnn()
        self.dec_rnn = DecoderRnn()
        self.attn = Attention()

    def call(self, input_frame, encoder_outputs, states, training=False):
        attn_rnn_state = states[0]
        dec_rnn_state = states[1]
        x = self.prenet(input_frame, training=training)
        x = keras.ops.expand_dims(x, axis=1)  # (B, 1, 128)
        q, s_a = self.attn_rnn(x, attn_rnn_state)
        q_squeezed = q[:, 0, :]  # (B, 256)
        c = self.attn(q_squeezed, encoder_outputs)
        c = keras.ops.expand_dims(c, axis=1)  # (B, 1, 128)
        q_exp = keras.ops.expand_dims(q_squeezed, axis=1)  # (B, 1, 256)
        dec_input = keras.ops.concatenate([c, q_exp], axis=-1)  # (B, 1, 384)
        x, s_d = self.dec_rnn(dec_input, dec_rnn_state)
        return x, (s_a, s_d)

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
        # inputs: (B, T, 80)
        x = self.input_proj(inputs)
        x = self.cbhg(x, training=training)
        x = self.output_proj(x)
        return x


class Tacotron(keras.Model):
    def __init__(self, num_freq=1025):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.post_net = PostNet(num_freq)

    def build(self, input_shape):
        # Eagerly build sub-layers so tf.function doesn't hit cuDNN shape-inference hangs
        dummy_text = keras.ops.zeros((1, 256, 256))
        encoder_seq = self.encoder(dummy_text)

        dummy_mel = keras.ops.zeros((1, 80))
        initial_states = self.decoder.get_initial_states(1)
        _ = self.decoder(dummy_mel, encoder_seq, initial_states)

        dummy_mels = keras.ops.zeros((1, 10, 80))
        _ = self.post_net(dummy_mels)

        self.built = True

    def call(self, inputs, training=False):
        # Dict inputs during training. Tensor inputs during inference
        if training:
            text_seq = inputs["text"]
            target_mels = inputs["mels"]
        else:
            text_seq = inputs
            target_mels = None

        encoder_seq = self.encoder(text_seq, training=training)

        batch_size = keras.ops.shape(text_seq)[0]
        r = self.decoder.dec_rnn.r

        if target_mels is not None:
            target_len = target_mels.shape[1]
            max_len = target_len // r + 1
        else:
            target_len = None
            max_len = 256

        next_mels = keras.ops.zeros(
            (batch_size, target_mels.shape[-1] if target_mels is not None else 80)
        )
        states = self.decoder.get_initial_states(batch_size)

        # Precompute teacher-forcing schedule so we can use a TF while_loop
        if training and target_mels is not None:
            indices = tf.range(max_len)
            idxs = tf.minimum((indices + 1) * r - 1, target_len - 1)
            teacher_inputs = tf.gather(target_mels, idxs, axis=1)
            teacher_inputs = tf.transpose(teacher_inputs, perm=[1, 0, 2])
        else:
            teacher_inputs = None

        output_ta = tf.TensorArray(
            tf.float32,
            size=max_len,
            element_shape=(None, r, 80),
        )

        def body(t, states, next_mels, output_ta):
            output_mel, new_states = self.decoder(next_mels, encoder_seq, states, training=training)
            output_ta = output_ta.write(t, output_mel)
            if training and teacher_inputs is not None:
                next_mels = teacher_inputs[t]
            else:
                next_mels = output_mel[:, -1, :]
            return t + 1, new_states, next_mels, output_ta

        t0 = tf.constant(0)
        _, _, _, output_ta = tf.while_loop(
            cond=lambda t, *_: t < max_len,
            body=body,
            loop_vars=[t0, states, next_mels, output_ta],
        )

        output_mels = output_ta.stack()  # (max_len, B, r, 80)
        output_mels = tf.transpose(output_mels, perm=[1, 0, 2, 3])  # (B, max_len, r, 80)
        output_mels = tf.reshape(output_mels, (batch_size, max_len * r, 80))

        if target_len is not None:
            # Truncate any over-generated frames so shapes match targets exactly
            output_mels = output_mels[:, :target_len, :]

        # Apply PostNet to the full sequence
        output_spectra = self.post_net(output_mels, training=training)

        return {"mels": output_mels, "spectra": output_spectra}

    def loss(self, target_mel, pred_mel, target_spectra, pred_spectra):
        mae = keras.losses.MeanAbsoluteError()
        seq2seq = mae(target_mel, pred_mel)
        spectral = mae(target_spectra, pred_spectra)
        return seq2seq + spectral  # type: ignore


# TODO Make sure loss is correct
# TODO Masking?
