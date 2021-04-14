# hai-q
Use a (Q)LSTM to generate short poetry (haiku)

Steps:

```
pip install -r requirements.txt
pip install -e .
./scripts/download_haiku.py
./scripts/train_haiku_model.py -Q 4 -B default.qubit.tf -D 'backprop'
./scripts/generate_haiku.py
```

Optional arguments:
```
usage: train_haiku_model.py [-h] [-e EPOCHS] [-b BATCH_SIZE] [-W WINDOW_SIZE] [-E EMBED_DIM] [-H HIDDEN_DIM] [-Q N_QUBITS] [-B BACKEND] [-D DIFF_METHOD] [-S SHOTS]

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCHS, --epochs EPOCHS
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -W WINDOW_SIZE, --window_size WINDOW_SIZE
  -E EMBED_DIM, --embed_dim EMBED_DIM
  -H HIDDEN_DIM, --hidden_dim HIDDEN_DIM
  -Q N_QUBITS, --n_qubits N_QUBITS
  -B BACKEND, --backend BACKEND
  -D DIFF_METHOD, --diff_method DIFF_METHOD
  -S SHOTS, --shots SHOTS
```