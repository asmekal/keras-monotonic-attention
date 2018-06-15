# keras-monotonic-attention
seq2seq attention in keras

AttentionDecoder class is modified version of the one here https://github.com/datalogue/keras-attention

The main differences:
* internal embedding for output layers
* Luong-style monotonic attention (optional)
* attention weight regularization (optional)
* teacher forcing
* fixing minor bugs like https://github.com/datalogue/keras-attention/issues/30

## Simple example

```python
model = Sequential()
model.add(Embedding(INPUT_VOCABE_SIZE, INPUT_EMBEDDING_DIM))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(AttentionDecoder(150, OUTPUT_VOCABE_SIZE, embedding_dim=OUTPUT_EMBEDDING_DIM))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
```

See sequential_example.py for details

## Attention mechanisms

To use Bahdanau [1] attention mechanism set `is_monotonic=False` and
`normalize_energy=False`. Actually it is also default mode.

```python
bahdanau_attention_decoder = AttentionDecoder(units, output_vocabe_size,
                                              is_monotonic=False,
                                              normalize_energy=False)
```

For Luong-style [2] attention set `is_monotonic=True` and
`normalize_energy=True`. Note that it is still additive, not multiplicative.

```python
luong_attention_decoder = AttentionDecoder(units, output_vocabe_size,
                                           is_monotonic=True,
                                           normalize_energy=True)
```

## Teacher forcing

To enable teacher forcing use that way:

```python
inputs = Input(shape=(None,), dtype='int64')
# encoder part
x = encoder_model(inputs)
# input that represent correct output sequence
labels_true = Input(shape=(None,), dtype='int64')

decoder_layer = AttentionDecoder(hidden_units, output_vocabe_size)

output = decoder_layer([x, labels_true])
```

See example.py for more details

## Position embedding

One additional feature in this repo is PositionEmbedding. It is embedding
for positional information. Just sin/cos ways with exponentionally. See [3], 
section 3.5, Position Encoding.

The layer should be used to extract some position information

## References

[1] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
"Neural machine translation by jointly learning to align and translate."
arXiv preprint arXiv:1409.0473 (2014).

[2] Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglass Eck
"Online and Linear-Time Attention by Enforcing Monotonic Alignments"
arXiv arXiv:1704.00784 (2017)

[3] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, 
Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
"Attention is all you need" arXiv arXiv:1706.03762
