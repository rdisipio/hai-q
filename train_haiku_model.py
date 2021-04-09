#!/usr/bin/env python

import pickle

import numpy as np
import pandas as pd

#from nltk import ngrams
from sklearn.model_selection import train_test_split 

import tensorflow as tf
from tensorflow import keras

from model import QLSTM, HaikuLM


def get_ngrams(X, n):
    ngrams = [X[i:i+n] for i in range(len(X)-n+1)]
    return ngrams


if __name__ == '__main__':
    EPOCHS = 10
    BATCH_SIZE = 32
    WINDOW_SIZE = 10
    EMBED_DIM = 4
    HIDDEN_DIM = 8
    N_QUBITS = 0
    BACKEND = 'default.qubits'

    f = open('haikus.pkl', 'rb')
    data = pickle.load(f)

    df = data['df']
    vocab = data['vocab']
    VOCAB_SIZE = len(vocab)
    print(f"Vocabulary size: {VOCAB_SIZE}")

    all_entries = df['token_ids']
    dataset = []
    for row in all_entries:
        ngrams = get_ngrams(row, WINDOW_SIZE)
        dataset.extend(ngrams)
    dataset = np.array(dataset, dtype=np.int64)
    
    X = dataset[:, :WINDOW_SIZE-1]
    y = dataset[:, -1]

    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    model = HaikuLM(
        embed_dim=EMBED_DIM,
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        n_qubits=N_QUBITS,
        backend=BACKEND)
    
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer, loss=keras.losses.SparseCategoricalCrossentropy())

    model.fit(X_train, y_train, 
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                shuffle=True)

    tf.model.save("model_haiku")

    model_params = {
            'embed_dim': EMBED_DIM,
            'vocab_size': VOCAB_SIZE,
            'hidden_dim': HIDDEN_DIM,
            'n_qubits': N_QUBITS
        }

