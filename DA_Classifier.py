from keras import Sequential
from keras.models import Model, Input
from keras.layers import (
    LSTM,
    GRU,
    Embedding,
    Dense,
    TimeDistributed,
    Dropout,
    Bidirectional,
)

from tf2crf import CRF
import config


class BiRNN_CRF_Model:
    def __init__(self, embedding_matrix, n_tags, rnn_type):
        self.embedding_matrix = embedding_matrix
        self.n_tags = n_tags                                    # Number of Dialogue Acts
        self.rnn = LSTM if rnn_type == "lstm" else GRU          # Choice of cell structure

    def get_model(self):
        return get_BiRNN_CRF_model(self.embedding_matrix, self.n_tags, self.rnn)



def get_BiRNN_CRF_model(embedding_matrix, n_tags, rnn, verbose=False):
    """
    Function which instantiates the Bi-directional RNN CRF model.

    embedding_matrix:
    n_tags:             The number of possible Dialogue Acts in the given corpus
    rnn:                Option to specify whether LSTM or GRU cells will be used
    verbose:            True to print info

    Note this is a modernised/ updated version of the model introduced by Kumar et al in 2017.
    """
    print("Loading model...")

    # Load key params from configuration file
    max_nr_utterances = config.data["max_nr_utterances"]
    max_nr_words = config.data["max_nr_words"]
    dropout_rate = config.model["dropout_rate"]             # Fraction of Neurons disabled during training
    nr_lstm_cells = config.model["nr_lstm_cells"]           # Dimensionality of output space from each cell

    # Define final (CRF) layer
    crf = CRF(dtype="float32")

    # Define Embedding Layer
    embedding_layer = Embedding(
        embedding_matrix.shape[0],
        embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=max_nr_words,
        trainable=True)

    # Build Utterance Encoder
    utterance_encoder = Sequential()
    utterance_encoder.add(embedding_layer)
    utterance_encoder.add(Bidirectional(rnn(nr_lstm_cells)))    # Combine word vectors to utterance vector
    utterance_encoder.add(Dropout(dropout_rate))                # Avoid over-fitting
    # utterance_encoder.add(Flatten())

    if verbose:
        utterance_encoder.summary()

    x_input = Input(shape=(max_nr_utterances, max_nr_words))
    h = TimeDistributed(utterance_encoder)(x_input)
    h = Bidirectional(rnn(nr_lstm_cells, return_sequences=True))(h)
    h = Dropout(dropout_rate)(h)
    h = Dense(n_tags, activation=None)(h)
    crf_output = crf(h)

    model = Model(x_input, crf_output)

    if verbose:
        model.summary()

    model.compile("adam", loss=crf.loss, metrics=[crf.accuracy])

    print("Done!")
    return model
