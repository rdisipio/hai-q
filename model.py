import pennylane as qml
import tensorflow as tf
from tensorflow import keras

class QLSTM(keras.layers.Layer):
    def __init__(self,
                units: int, 
                n_qubits: int,
                n_qlayers: int=1,
                return_sequences=False, 
                return_state=False,
                backend="default.qubit"):
        super(QLSTM, self).__init__()
        self.units = units
        self.concat_size = None
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.return_sequences = return_sequences
        self.return_state = return_state

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        def _circuit_forget(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_forget)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]
        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="tf")

        def _circuit_input(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_input)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_input)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]
        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="tf")

        def _circuit_update(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]
        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="tf")

        def _circuit_output(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_output)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]
        self.qlayer_output = qml.QNode(_circuit_output, self.dev_output, interface="tf")
        
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
            #h_t = h_t[0] #?
            #c_t = c_t[0] #?
        
        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = inputs[:, t, :]

            # Concatenate input and hidden state
            v_t = tf.concat((h_t, x_t), axis=1)
        
            # match qubit dimension
            y_t = tf.matmul(v_t, self.W_in)

            f_t = tf.math.sigmoid(tf.matmul(
                tf.dtypes.cast(self.VQC['forget'](y_t), tf.float32), self.W_out))
            i_t = tf.math.sigmoid(tf.matmul(
                tf.dtypes.cast(self.VQC['input'](y_t), tf.float32), self.W_out))
            g_t = tf.math.sigmoid(tf.matmul(
                tf.dtypes.cast(self.VQC['update'](y_t), tf.float32), self.W_out))
            o_t = tf.math.sigmoid(tf.matmul(
                tf.dtypes.cast(self.VQC['output'](y_t), tf.float32), self.W_out))

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * tf.math.tanh(c_t)
            hidden_seq.append(h_t)
        hidden_seq = tf.Tensor(hidden_seq)
        hidden_seq = hidden_seq.transpose(0,1)

        if self.return_sequences is True:
            return hidden_seq
        else:
            return hidden_seq[-1]


class HaikuLM(tf.keras.Model):
    def __init__(self,
                embed_dim: int,
                vocab_size: int,
                hidden_dim: int,
                n_qubits: int=0,
                backend: str='default.qubit',
                **kwargs):
        super(HaikuLM, self).__init__(**kwargs)
    
        self.embed = keras.layers.Embedding(vocab_size, embed_dim)
        if n_qubits == 0:
            self.lstm = keras.layers.LSTM(hidden_dim)
        else:
            self.lstm = QLSTM(hidden_dim, n_qubits=n_qubits, backend=backend)
        self.hidden2id = keras.layers.Dense(vocab_size, activation='softmax')
    
    def call(self, inputs):
        x = self.embed(inputs)
        logits = self.lstm(x)
        probs = self.hidden2id(logits)
        return probs
    