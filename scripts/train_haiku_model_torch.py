#!/usr/bin/env python

import argparse
import pickle
import time

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from jax.config import config
config.update("jax_enable_x64", True)

#from nltk import ngrams
from sklearn.model_selection import train_test_split 

from haiq.model_torch import QLSTM, HaikuLM

MAX_SEQ_LEN = 256


def get_ngrams(X, n):
    ngrams = [X[i:i+n] for i in range(len(X)-n+1)]
    return ngrams


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    print(f"Number of training batches: {len(iterator)}")
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    for i, batch in enumerate(iterator):
        print(f"Batch {i+1} / {len(iterator)}")
        optimizer.zero_grad()

        inputs, labels = batch

        inputs = torch.LongTensor(inputs)
        if inputs.size(1) > MAX_SEQ_LEN:
            inputs = inputs[:, :MAX_SEQ_LEN]
        predictions = model(inputs)#.squeeze(0)
        loss = criterion(predictions, labels)
          
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        #epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            inputs, labels = batch

            inputs = torch.LongTensor(inputs)
            if inputs.size(1) > MAX_SEQ_LEN:
                inputs = inputs[:, :MAX_SEQ_LEN]
            predictions = model(inputs)#.squeeze(1)
            loss = criterion(predictions, labels)
            
            epoch_loss += loss.item()
            #epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-l', '--lr', type=float, default=0.01)
    parser.add_argument('-W', '--window_size', type=int, default=10)
    parser.add_argument('-E', '--embed_dim', type=int, default=4)
    parser.add_argument('-H', '--hidden_dim', type=int, default=8)
    parser.add_argument('-Q', '--n_qubits', type=int, default=4)
    parser.add_argument('-B', '--backend', default='default.qubit.autograd')
    parser.add_argument('-I', '--interface', default='torch')
    parser.add_argument('-D', '--diff_method', default='backprop')
    parser.add_argument('-S', '--shots', type=int, default=None)
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    WINDOW_SIZE = args.window_size
    EMBED_DIM = args.embed_dim
    HIDDEN_DIM = args.hidden_dim
    N_QUBITS = args.n_qubits
    INTERFACE = args.interface
    BACKEND = args.backend
    DIFF_METHOD = args.diff_method
    SHOTS = args.shots

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

    data_train = [(x,y) for x,y in zip(X_train, y_train)]
    ds_train = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    data_test = [(x,y) for x,y in zip(X_test, y_test)]
    ds_test = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False)


    model = HaikuLM(
        embed_dim=EMBED_DIM,
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        n_qubits=N_QUBITS,
        interface=INTERFACE,
        backend=BACKEND,
        diff_method=DIFF_METHOD)

    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
    criterion = torch.nn.CrossEntropyLoss()  # logits -> log_softmax -> NLLloss 
    
    history = {
        'loss': []
    }
    # training loop
    best_valid_loss = float('inf')
    for iepoch in range(EPOCHS):
        start_time = time.time()
        print(f"Epoch {iepoch+1}/{EPOCHS}")
        train_loss, train_acc = train(model, ds_train, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, ds_test, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')
        
        print(f'Epoch: {iepoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')