import pennylane as qml
import tensorflow as tf
from tensorflow import keras

import numpy as np

from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)


class CLSTM(keras.layers.Layer):
    def __init__(self,
                units: int, 
                return_sequences=False, 
                return_state=False):
        super(CLSTM, self).__init__()
        self.units = units
        self.concat_size = None
        self.return_sequences = return_sequences
        self.return_state = return_state

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.U = self.add_weight(shape=(input_dim, self.units * 4))
        self.W = self.add_weight(shape=(self.units, self.units * 4))
        self.b = self.add_weight(shape=(self.units * 4,))
        
    def call(self, inputs, initial_state=None):
        batch_size, seq_length, features_size = tf.shape(inputs)

        hidden_seq = []
        if initial_state is None:
            h_t = tf.zeros((batch_size, 4 * self.units))  # hidden state (output)
            c_t = tf.zeros((batch_size, 4 * self.units))  # cell state
        else:
            h_t, c_t = initial_state
        
        z = tf.matmul(inputs, self.W)
        z += tf.matmul(h_t, self.U)
        z += self.b

        z0, z1, z2, z3 = tf.split(z, num_or_size_splits=4, axis=1)
        i_t = tf.math.sigmoid(z0)
        f_t = tf.math.sigmoid(z1)
        g_t = tf.math.sigmoid(z2)
        o_t = tf.math.sigmoid(z3)
        c_t = (f_t * c_t) + (i_t * g_t)
        h_t = o_t * tf.math.tanh(c_t)


        ##############

        x_t = tf.transpose(inputs, [1,0,2])  # (seq_len, batch_size, features_size)
        gates = x_t @ self.W + h_t @ self.U + self.b
        i_t, f_t, g_t, o_t = tf.split()
                tf.math.sigmoid(gates[:, :, :HS]),
                tf.math.sigmoid(gates[:, :, HS:HS*2]),
                tf.math.sigmoid(gates[:, :, HS*2:HS*3]),
                tf.math.sigmoid(gates[:, :, HS*3:]),
            )
        c_t = (f_t * c_t) + (i_t * g_t)
        h_t = o_t * tf.math.tanh(c_t)

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = inputs[:, t, :]

            gates = x_t @ self.W + h_t @ self.U + self.b

            HS = self.units
            i_t, f_t, g_t, o_t = (
                tf.math.sigmoid(gates[:, :HS]),
                tf.math.sigmoid(gates[:, HS:HS*2]),
                tf.math.sigmoid(gates[:, HS*2:HS*3]),
                tf.math.sigmoid(gates[:, HS*3:]),
            )
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * tf.math.tanh(c_t)
            hidden_seq.append(h_t)
        hidden_seq = tf.convert_to_tensor(hidden_seq)  # (seq, batch, embed)
        hidden_seq = tf.transpose(hidden_seq, (1,0,2)) # (batch, seq, embed)

        if self.return_sequences is True:
            return hidden_seq
        else:
            return hidden_seq[:, -1, :]


class QLSTM(keras.layers.Layer):
    def __init__(self,
                units: int, 
                n_qubits: int=4,
                n_qlayers: int=1,
                return_sequences=False, 
                return_state=False,
                backend="default.qubit.tf",
                diff_method='backprop',
                interface='tf',
                shots=0):
        super(QLSTM, self).__init__()
        self.units = units
        self.concat_size = None
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.interface = interface  # 'jax', 'tf'
        self.backend = backend  # "default.qubit.tf", "qiskit.basicaer", "qiskit.ibm"
        self.diff_method = diff_method  # "backprop", "adjoint"

        self.return_sequences = return_sequences
        self.return_state = return_state

        self.wires = [i for i in range(self.n_qubits)]

        if 'qulacs' in self.backend:
            print("Using qulacs simulator as backend")
            self.device = qml.device(self.backend, wires=self.wires, gpu=True) #, shots=shots)
        self.device = qml.device(self.backend, wires=self.wires) #, shots=shots)
        print(f"Backend: {self.backend}")
        print(f"Differentiation method: {self.diff_method}")

        def _circuit_forget(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires]
        self.qlayer_forget = qml.QNode(_circuit_forget, self.device, interface=self.interface, diff_method=self.diff_method)

        def _circuit_input(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires]
        self.qlayer_input = qml.QNode(_circuit_input, self.device, interface=self.interface, diff_method=self.diff_method)

        def _circuit_update(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires]
        self.qlayer_update = qml.QNode(_circuit_update, self.device, interface=self.interface, diff_method=self.diff_method)

        def _circuit_output(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires]
        self.qlayer_output = qml.QNode(_circuit_output, self.device, interface=self.interface, diff_method=self.diff_method)
        
        weight_shapes = {"weights": (self.n_qlayers, self.n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({self.n_qlayers}, {self.n_qubits})")

        self.VQC = {
            'forget': qml.qnn.KerasLayer(self.qlayer_forget, weight_shapes, output_dim=self.n_qubits),
            'input': qml.qnn.KerasLayer(self.qlayer_input, weight_shapes, output_dim=self.n_qubits),
            'update': qml.qnn.KerasLayer(self.qlayer_update, weight_shapes, output_dim=self.n_qubits),
            'output': qml.qnn.KerasLayer(self.qlayer_output, weight_shapes, output_dim=self.n_qubits)
        }

    def build(self, input_shape):
        self.concat_size = input_shape[-1] + self.units
        self.W_in = self.add_weight(shape=(self.concat_size, self.n_qubits),
            initializer='glorot_uniform', trainable=True)
        self.W_out = self.add_weight(shape=(self.n_qubits, self.units),
            initializer='glorot_uniform', trainable=True)
        
    def call(self, inputs, initial_state=None):
        batch_size, seq_length, features_size = tf.shape(inputs)

        hidden_seq = []
        if initial_state is None:
            h_t = tf.zeros((batch_size, self.units))  # hidden state (output)
            c_t = tf.zeros((batch_size, self.units))  # cell state
        else:
            h_t, c_t = initial_state
        
        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = inputs[:, t, :]

            # Concatenate input and hidden state
            v_t = tf.concat((h_t, x_t), axis=1)
        
            # match qubit dimension
            y_t = tf.matmul(v_t, self.W_in)
            y_t = tf.cast(y_t, tf.float64)
            z_forget = tf.cast(self.VQC['forget'](y_t), tf.float32)
            z_input = tf.cast(self.VQC['input'](y_t), tf.float32)
            z_update = tf.cast(self.VQC['update'](y_t), tf.float32)
            z_output = tf.cast(self.VQC['output'](y_t), tf.float32)

            f_t = tf.math.sigmoid(tf.matmul(z_forget, self.W_out))
            i_t = tf.math.sigmoid(tf.matmul(z_input, self.W_out))
            g_t = tf.math.sigmoid(tf.matmul(z_update, self.W_out))
            o_t = tf.math.sigmoid(tf.matmul(z_output, self.W_out))

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * tf.math.tanh(c_t)
            hidden_seq.append(h_t)
        hidden_seq = tf.convert_to_tensor(hidden_seq)  # (seq, batch, embed)
        hidden_seq = tf.transpose(hidden_seq, (1,0,2)) # (batch, seq, embed)

        if self.return_sequences is True:
            return hidden_seq
        else:
            return hidden_seq[:, -1, :]


class HaikuLM(tf.keras.Model):
    def __init__(self,
                embed_dim: int,
                vocab_size: int,
                hidden_dim: int,
                n_qubits: int=4,
                backend: str='default.qubit.tf',
                diff_method='backprop',
                shots=0,
                **kwargs):
        super(HaikuLM, self).__init__(**kwargs)
    
        self.embed = keras.layers.Embedding(vocab_size, embed_dim)
        if n_qubits is None:
            self.lstm = keras.layers.LSTM(hidden_dim)
        elif n_qubits == 0:
            print(f"Using classical LSTM with hidden dim = {hidden_dim}")
            self.lstm = CLSTM(hidden_dim)
        else:
            print(f"Using quantum LSTM with hidden dim = {hidden_dim} and {n_qubits} qubits")
            self.lstm = QLSTM(hidden_dim, n_qubits=n_qubits, 
                            backend=backend,
                            diff_method=diff_method,
                            shots=shots)
        self.hidden2id = keras.layers.Dense(vocab_size, activation='softmax')
    
    def call(self, inputs):
        x = self.embed(inputs)
        logits = self.lstm(x)
        probs = self.hidden2id(logits)
        return probs
    