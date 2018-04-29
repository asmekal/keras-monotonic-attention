from keras import backend as K
from keras.layers import Embedding, Layer
import numpy as np


def pos_encoding(time, n_waves, d_model=100):
    encodings = np.zeros((2 * n_waves, time))
    positions = np.arange(time, dtype=np.float32)
    for i in range(n_waves):
        encodings[2 * i] = np.sin(positions / 10. ** (2. * i / d_model))
        encodings[2 * i + 1] = np.cos(positions / 10. ** (2. * i / d_model))
    return np.transpose(encodings)  # shape (time, n_waves)


class PositionEmbedding(Layer):

    def __init__(self, max_time=1000, n_waves=16, d_model=64, name='PositionEmbedding', **kwargs):
        """
        Position embedding via sin and cos functions
        For incoming ``position`` produces embedding of dimension ``n_waves * 2``
        ``embedding[2*i] = sin(positions / 10. ** (2. * i / d_model))``
        ``embedding[2*i+1] = cos(positions / 10. ** (2. * i / d_model))``
        :param max_time: maximum time dimension of input sequence
        """
        self.max_time = max_time
        self.n_waves = n_waves
        self.d_model = d_model
        emb_weights = pos_encoding(max_time, n_waves, d_model=d_model)
        self.embedding_layer = Embedding(max_time, n_waves * 2,
                                         weights=[emb_weights],
                                         trainable=False)

        super(PositionEmbedding, self).__init__(**kwargs)
        self.name = name

    def build(self, input_shapes):
        self.embedding_layer.build((None, None))
        self.built = True

    def call(self, x):
        samples = K.shape(x)[0]
        time = K.shape(x)[1]
        pos_enc = self.embedding_layer(
            K.reshape(K.arange(time, dtype='int32'), (1, -1)))
        return K.tile(pos_enc, (samples, 1, 1))

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        return (None, None, self.n_waves * 2)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        # TODO: teacher forcing
        config = {
            'n_waves': self.n_waves,
            'max_time': self.max_time,
            'd_model': self.d_model
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))