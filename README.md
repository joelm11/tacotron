# Taco

A Tacotron2 implementation.

While the model trains and loss converges, attention scores are rarely linear during training runs. This is a known problem with Tacotron models. Modern TTS models don't use the autoregressive techniques employed herein.