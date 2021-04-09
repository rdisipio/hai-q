#!/usr/bin/env python

import pickle
import numpy as np
import torch

from train_haiku_model import HaikuLM

if __name__ == '__main__':
    model_path = "haiku_model.pt"
    weights, params = torch.load(model_path)
    backend = 'default.qubit'

    f = open('haikus.pkl', 'rb')
    data = pickle.load(f)
    vocab = data['vocab']
    VOCAB_SIZE = len(vocab)
    print(f"Vocabulary size: {VOCAB_SIZE}")
    int2char = {i:c for i,c in enumerate(vocab)}
    char2int = {c:i for i,c in enumerate(vocab)}

    model = HaikuLM(embed_dim=params['embed_dim'],
        vocab_size=params['vocab_size'],
        hidden_dim=params['hidden_dim'],
        n_qubits=params['n_qubits'],
        backend=backend)
    model.load_state_dict(weights)
    model.eval()

    # generate characters
    token_ids = torch.LongTensor([char2int['[sos]']])
    for i in range(10):
        logits = model.forward(token_ids.unsqueeze(0))
        id = torch.argmax(logits, dim=-1)
        token_ids = torch.cat([token_ids, id])
        #if id.item() == char2int['eos']:
        #    break
    token_ids = token_ids.tolist()
    print(token_ids)

    chars = [int2char[id] for id in token_ids]
    haiku = "".join(chars)
    print(haiku)

