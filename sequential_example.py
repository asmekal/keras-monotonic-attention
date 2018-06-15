"""
Toy example - reconstructs input sequence from itself
Simple sequential model
"""

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from attention_decoder import AttentionDecoder

import numpy as np

INPUT_VOCABE_SIZE = 50
# in this examle input sequence is the same as output sequence
OUTPUT_VOCABE_SIZE = 50

INPUT_EMBEDDING_DIM = 10
OUTPUT_EMBEDDING_DIM = 10

model = Sequential()
model.add(Embedding(INPUT_VOCABE_SIZE, INPUT_EMBEDDING_DIM))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(AttentionDecoder(150, OUTPUT_VOCABE_SIZE, embedding_dim=OUTPUT_EMBEDDING_DIM))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

n = 10000
t = 10
x = np.random.randint(0, INPUT_VOCABE_SIZE, size=(n, t))
# reshape is needed for computing sparse_categorical_crossentropy loss
# which expect labels_true to have shape (batch, time, 1) and not (batch, time)
y = np.expand_dims(x, -1)

model.fit(x, y, epochs=10)
