import keras
from keras import layers
from keras.src import backend, ops, tree


def make_encoder():
    # inputs: Batches of text (B, T)
    # outputs: Encoded features (B, T, 512)
    encoder_input = keras.Input(shape=(None,), dtype="int32", name="encoder_input")  # chars as ints

    emb = layers.Embedding(input_dim=256, output_dim=512)(encoder_input)
    conv = layers.Conv1D(512, kernel_size=5, padding="same")(emb)
    conv = layers.Conv1D(512, kernel_size=5, padding="same")(conv)
    conv = layers.Conv1D(512, kernel_size=5, padding="same")(conv)
    norm = layers.BatchNormalization()(conv)  # TODO Last axis?
    dense = layers.Dense(256, activation="relu")(norm)
    encoder_output = layers.Bidirectional(layers.LSTM(256))(dense)

    encoder = keras.Model(inputs=[encoder_input], outputs=encoder_output)

    return encoder


class LocationSensitiveAttention(keras.layers.Layer):
    def __init__(self, units=128, n_filters=32, n_taps=31):
        super().__init__()
        self.units = 128
        self.p_q = layers.Dense(units, activation="linear", use_bias=False)
        self.p_k = layers.Dense(units, activation="linear", use_bias=False)
        self.p_l = layers.Dense(units, activation="linear", use_bias=False)
        self.loc_convs = layers.Conv1D(filters=n_filters, kernel_size=n_taps, padding="same")
        self.v_a = keras.layers.Dense(1, use_bias=False)  # Attn scale

    def build(self, input_shape):
        # Build sublayers with their expected input shapes
        self.p_q.build((None, 1024))  # query from LSTM2 output
        self.p_k.build((None, 512))  # encoder_sequence features
        self.p_l.build((None, 32))  # location features (conv output)
        self.loc_convs.build((None, None, 1))  # cumulative weights (B, S, 1)
        self.v_a.build((None, 128))  # attention score features
        super().build(input_shape)

    def call(self, query, w_cum, encoder_sequence):
        # query: (B, D)
        # w_cum: (B, S)
        # encoder_sequence: (B, S, D_enc)

        # Location features from cumulative attn weights
        locs = self.loc_convs(keras.ops.expand_dims(w_cum, -1))

        # Projections
        p_locs = self.p_l(locs)  # (B, S, U)
        p_q = self.p_q(query)  # (B, U)
        p_k = self.p_k(encoder_sequence)  # (B, S, U)

        # Attn scores and weights
        scores = keras.ops.tanh(p_q[:, None, :] + p_k + p_locs)  # (B, S, U)
        scores = self.v_a(scores)[..., 0]
        weights = keras.ops.softmax(scores)  # (B, S)

        # Generate context vector
        c = keras.ops.einsum("bs,bsd->bd", weights, p_k)

        return c, w_cum + weights, weights


class DecoderCell(keras.layers.Layer):
    def __init__(self, units=256, **kwargs):
        super().__init__(**kwargs)
        self.LSTM_UNITS = 1024
        self.N_MELS = 80
        # Prenet
        self.d1 = layers.Dense(units, activation="relu")
        self.d2 = layers.Dense(units, activation="relu")
        # Lstms
        self.l1 = layers.LSTMCell(self.LSTM_UNITS)
        self.l2 = layers.LSTMCell(self.LSTM_UNITS)
        # Attention
        self.attn = LocationSensitiveAttention()
        # Frame and stop projections
        self.df = layers.Dense(self.N_MELS, activation="linear")
        self.ds = layers.Dense(1, activation="sigmoid")
        # Concatenation layers
        self.c1 = layers.Concatenate()
        self.c2 = layers.Concatenate()

        # Declare state outputs
        self.state_size = (
            self.LSTM_UNITS,
            self.LSTM_UNITS,
            self.LSTM_UNITS,
            self.LSTM_UNITS,
            100,  # Max S
            128,  # D default maybe 512?
        )
        # Declare other outputs
        self.output_size = self.N_MELS + 1

    def build(self, input_shape):
        # input_shape is (batch_size, 80) for a single mel frame
        self.d1.build(input_shape)
        self.d2.build((None, self.d1.units))

        # After prenet (256) + context c (128)
        self.l1.build((None, self.d1.units + 128))
        self.l2.build((None, self.LSTM_UNITS))

        # After LSTM2 (1024) + context c (128)
        self.df.build((None, self.LSTM_UNITS + 128))
        self.ds.build((None, self.LSTM_UNITS + 128))

        # Build attention module
        self.attn.build(None)

        super().build(input_shape)

    def call(self, target_mel, states, encoder_sequence=None):
        h1, c1, h2, c2, attn_w, c = states

        x = self.d1(target_mel)
        x = self.d2(x)
        x = self.c1([x, c])
        x, (h1_n, c1_n) = self.l1(x, [h1, c1])
        x, (h2_n, c2_n) = self.l2(x, [h2, c2])
        c_n, attn_w_n, _ = self.attn(x, attn_w, encoder_sequence=encoder_sequence)

        # Project outputs (frame and stop)
        x = self.c2([x, c])
        frame = self.df(x)
        stop = self.ds(x)

        output = layers.Concatenate()([frame, stop])
        return output, (h1_n, c1_n, h2_n, c2_n, attn_w_n, c_n)


class DecoderRNN(layers.RNN):
    def call(self, sequences, encoder_sequence=None, initial_state=None, mask=None, training=False):
        self._encoder_sequence = encoder_sequence
        return super().call(
            sequences,
            initial_state=initial_state,
            mask=mask,
            training=training,
        )

    def inner_loop(self, sequences, initial_state, mask, training=False):
        cell_kwargs = {}
        if isinstance(self.cell, layers.Layer) and self.cell._call_has_training_arg:
            cell_kwargs["training"] = training

        def step(inputs, states):
            if backend.backend() == "torch" and self.stateful:
                states = tree.map_structure(ops.copy, states)
            output, new_states = self.cell(
                inputs, states, encoder_sequence=self._encoder_sequence, **cell_kwargs
            )
            if not tree.is_nested(new_states):
                new_states = [new_states]
            return output, new_states

        if not tree.is_nested(initial_state):
            initial_state = [initial_state]

        return backend.rnn(
            step,
            sequences,
            initial_state,
            go_backwards=self.go_backwards,
            mask=mask,
            unroll=self.unroll,
            input_length=sequences.shape[1],
            zero_output_for_mask=self.zero_output_for_mask,
            return_all_outputs=self.return_sequences,
        )


def make_decoder():
    # inputs: Encoded text (B, T, 512), Target mels (B, Tm, 80), Target stop (B, Tm, 1)
    # inputs: Length masks (B, Tm, 1)
    # outputs: Mel frames (B, Tm, 80), Stop tokens (B, Tm, 1)
    enc_text = keras.Input(shape=(None, 512), name="encoder_sequence")
    target_mels = keras.Input(shape=(None, 80), name="target_mels")
    # target_stops = keras.Input(shape=(None, 1), name="target_stop")
    len_masks = keras.Input(shape=(None, 1), name="len_masks")

    # Need to handle initial states for the cell
    # Need to make sure wrapping will propogate states correctly
    # xf, xs, states = DecoderCell(enc_text)(target_mels, None)
    cell = DecoderCell()
    rnn_layer = DecoderRNN(cell, return_sequences=True)
    outputs = rnn_layer(target_mels, encoder_sequence=enc_text)

    xf = outputs[:, :, :80]
    xs = outputs[:, :, 80:]

    decoder = keras.Model(inputs=[enc_text, target_mels, len_masks], outputs=[xf, xs])

    return decoder


def make_post_net():
    return


def make_wave_net():
    # Do eventually
    return


def main():
    # encoder = make_encoder()
    # encoder.summary()
    decoder = make_decoder()
    decoder.summary()


if __name__ == "__main__":
    main()
