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

        ## Reshape mels, append <GO> frame, keep every `r`th frame
        mels = keras.ops.reshape(mels, (batch_size, n_frames // r, r * n_mels))
        mels = keras.ops.concatenate(
            [keras.ops.zeros((batch_size, n_frames // r, r * n_mels)), mels], axis=1
        )
        mels = mels[:, :, -n_mels:]

        zero_frame = keras.ops.zeros((keras.ops.shape(mels)[0], 1, r * n_mels))  # type: ignore
        mels = keras.ops.concatenate([zero_frame, mels], axis=1)

        enc_seq = self.encoder(text, training=training)  # (B, T, 256)

        pred_mels, _ = self.decoder(mels, enc_seq, None)  # (B, T_dec, r * n_mels)

        pred_spectra = self.post_net(pred_mels)  # (B, T, n_freq)

        # Reshape outputs for loss calculation
        return {"pred_mels": pred_mels, "pred_spectra": pred_spectra}
