#!/usr/bin/env python

import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras 

from model import HaikuLM, QLSTM

#from train_haiku_model import HaikuLM

if __name__ == '__main__':
    model_path = "model_haiku"
    backend = 'default.qubit'
    #model = tf.keras.models.load_model(model_path, 
    #                custom_objects={'HaikuLM': HaikuLM, 'QLSTM': QLSTM})

    f = open('haikus.pkl', 'rb')
    data = pickle.load(f)
    vocab = data['vocab']
    VOCAB_SIZE = len(vocab)
    print(f"Vocabulary size: {VOCAB_SIZE}")
    int2char = {i:c for i,c in enumerate(vocab)}
    char2int = {c:i for i,c in enumerate(vocab)}

    model = HaikuLM(4, VOCAB_SIZE, 8, 0)
    model.load_weights(model_path)

    # generate characters
    token_ids = tf.convert_to_tensor([char2int['[sos]']], dtype=tf.int64)
    for i in range(10):
        logits = model(tf.expand_dims(token_ids, axis=0))
        id = tf.argmax(logits, axis=-1)
        token_ids = tf.concat([token_ids, id], axis=-1)
        #if id.item() == char2int['eos']:
        #    break
    token_ids = token_ids.numpy()
    print(token_ids)

    chars = [int2char[id] for id in token_ids]
    haiku = "".join(chars)
    print(haiku)

