# keras-monotonic-attention
seq2seq attention in keras

AttentionDecoder class is modified version of the one here https://github.com/datalogue/keras-attention

The main differences:
* internal embedding for output layers
* Luong-style monotonic attention (optional)
* attention weight regularization (optional)
* teacher forcing
* fixing minor bugs like https://github.com/datalogue/keras-attention/issues/30
