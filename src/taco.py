import keras

from decoder import Decoder, PostNet
from encoder import Encoder


class Tts(keras.Model):
    def __init__(self, r=5, n_mels=80, num_freq=1025):
        super().__init__()
        self.r = r
        self.n_mels = n_mels
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.post_net = PostNet(r, n_mels, num_freq)

    def call(self, inputs, r=5, training=False):
        # inputs[0] (text): (B, T=256, D=256)
        # inputs[1] (mels): (B, n_frames=800, n_mels=80) teacher forcing and loss
        if training:
            text = inputs["text"]
            mels = inputs["mels"]
        else:
            return

        batch_size = keras.ops.shape(mels)[0]
        n_frames = keras.ops.shape(mels)[1]
        n_mels = mels.shape[2]

        # Group target mel frames by reduction factor r
        grouped = keras.ops.reshape(mels, (batch_size, n_frames // r, r, n_mels))
        # Teacher forcing input: last frame of each group, shifted right with zero <GO> frame
        dec_input_frames = grouped[:, :, -1, :]  # (B, T//r, n_mels)
        go_frame = keras.ops.zeros((batch_size, 1, n_mels))
        dec_input = keras.ops.concatenate([go_frame, dec_input_frames[:, :-1, :]], axis=1)

        enc_seq = self.encoder(text, training=training)  # (B, T_text, 256)

        pred_mels_grouped, _ = self.decoder(dec_input, enc_seq, None)  # (B, T//r, r * n_mels)

        # Reshape predictions back to full time resolution for loss
        pred_mels = keras.ops.reshape(pred_mels_grouped, (batch_size, n_frames, n_mels))

        pred_spectra = self.post_net(pred_mels_grouped)  # (B, T, n_freq)

        return {"pred_mels": pred_mels, "pred_spectra": pred_spectra}
