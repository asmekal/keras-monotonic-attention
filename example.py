"""
Toy example - reconstructs input sequence from itself
Shows usage of teacher forcing in AttentionDecoder
And PositionEmbedding
"""

from keras.layers import Input, Embedding, concatenate
from keras.models import Model
import numpy as np

from attention_decoder import AttentionDecoder
from position_embedding import PositionEmbedding

np.random.seed(0)

# generating data
n, t = 100000, 20
n_labels = 10
x = np.random.randint(0, n_labels, size=(n, t))
y = np.expand_dims(x, axis=-1)
x_val = np.random.randint(0, n_labels, size=(n // 100, t))
y_val = np.expand_dims(x_val, axis=-1)

# building model
inputs = Input(shape=(None,), dtype='int64')
outp_true = Input(shape=(None,), dtype='int64')
embedded = Embedding(n_labels, n_labels, weights=[np.eye(n_labels)], trainable=False)(inputs)

pos_emb = PositionEmbedding(max_time=1000, n_waves=20, d_model=40)(embedded)
nnet = concatenate([embedded, pos_emb], axis=-1)

attention_decoder = AttentionDecoder(40, n_labels,
                                     embedding_dim=5,
                                     is_monotonic=False,
                                     normalize_energy=False)
# use teacher forcing
output = attention_decoder([nnet, outp_true])
# (alternative) without teacher forcing
# output = attention_decoder(nnet)
# or
# output = attention_decoder([nnet, outp_true], use_teacher_forcing=False)
# the last variant is useful for generating outputs with number of timesteps different from input
# (without it the length of the output sequence will be the same as input sequence)
# so to produce outputs of different shape on inference the one could place outp_true=np.zeros(batch_size, outp_time)
model = Model(inputs=[inputs, outp_true], outputs=[output])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adadelta',
    metrics=['accuracy'])
model.summary()

model.fit([x, np.squeeze(y, axis=-1)], y,
          epochs=5,
          validation_data=([x_val, np.squeeze(y_val, axis=-1)], y_val))
