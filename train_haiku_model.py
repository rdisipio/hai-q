#!/usr/bin/env python

import argparse
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-W', '--window_size', type=int, default=10)
    parser.add_argument('-E', '--embed_dim', type=int, default=4)
    parser.add_argument('-H', '--hidden_dim', type=int, default=8)
    parser.add_argument('-Q', '--n_qubits', type=int, default=4)
    parser.add_argument('-B', '--backend', default='default.qubits')
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    WINDOW_SIZE = args.window_size
    EMBED_DIM = args.embed_dim
    HIDDEN_DIM = args.hidden_dim
    N_QUBITS = args.n_qubits
    BACKEND = args.backend

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

