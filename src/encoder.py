from keras import layers

from ops import Cbhg, PreNet


class Encoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.prenet = PreNet()
        self.cbhg = Cbhg(16, (128, 128), 128)

    def call(self, inputs, training=None):
        # inputs: (B, T, 256)
        # outputs: (B, T, 256)
        x = self.prenet(inputs, training=training)
        encoder_outputs = self.cbhg(x, training=training)
        return encoder_outputs
