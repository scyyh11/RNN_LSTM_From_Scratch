# 🧠 RNN & LSTM From Scratch (with PyTorch Comparison)

This project implements a **Recurrent Neural Network (RNN)** and **Long Short-Term Memory (LSTM)** **from scratch using NumPy**, trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit dataset.

---

## 📦 Features

- ✅ Manual forward pass and backpropagation through time (BPTT)
- ✅ Gate-level LSTM implementation: input, forget, output, cell candidate
- ✅ Gradient clipping and training loop
- ✅ PyTorch `DataLoader` integration for fast data batching
- ✅ Benchmarking: custom NumPy models vs PyTorch models

---

## 🗂️ Project Structure

```
├── main.py               # Entry point: trains both custom and PyTorch models
├── models/
│   ├── rnn.py            # Custom RNN class (NumPy)
│   ├── lstm.py           # Custom LSTM class (NumPy)
│   └── activations.py    # Custom sigmoid/tanh functions
├── data/                 # MNIST data (downloaded automatically)
└── README.md
```

---

## 🧪 Training Results (Example)

| Model         | Accuracy (5 epochs, 5k samples) |
|---------------|---------------------------------|
| Custom RNN    | 89.28%                          |
| PyTorch RNN   | N/A                             |
| Custom LSTM   | 91.12%                          |
| PyTorch LSTM  | 93.36%                          |

*Results may vary slightly depending on your machine.*

---

## 🚀 How to Run

1. **Install dependencies using pip:**

```bash
pip install -r requirements.txt
```

2. **Run the main script:**

```bash
python main.py
```

---
