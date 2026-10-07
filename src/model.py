import keras
from keras import layers


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
        self.loc_convs = layers.Conv1D(filters=n_filters, kernel_size=n_taps)
        self.v_a = keras.layers.Dense(1, use_bias=False)  # Attn scale

    def call(self, query, w_cum, encoder_sequence):
        # query: (B, D)
        # w_cum: (B, S)
        # encoder_sequence: (B, S, D_enc)

        # Location features from cumulative attn weights
        locs = self.loc_convs(w_cum)

        # Projections
        p_locs = self.p_l(locs[:, :, None])  # (B, S, U)
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
        self.encoder_sequence = None
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

    def call(self, target_mel, states, encoder_sequence=None):
        h1, c1, h2, c2, attn_w, c = states

        x = self.d1(target_mel)
        x = self.d2(x)
        x = self.c1([x, c])
        x, h1_n, c1_n = self.l1(x, initial_state=[h1, c1])
        x, h2_n, c2_n = self.l2(x, initial_state=[h2, c2])
        c_n, attn_w_n, _ = self.attn(x, attn_w, encoder_sequence=self.encoder_sequence)

        # Project outputs (frame and stop)
        x = self.c2([x, c])
        frame = self.df(x)
        stop = self.ds(x)

        output = layers.Concatenate()([frame, stop])
        return output, (h1_n, c1_n, h2_n, c2_n, attn_w_n, c_n)


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
    rnn_layer = layers.RNN(cell, return_sequences=True)
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
