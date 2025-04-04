import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models.rnn import RNN
from models.lstm import LSTM
import time
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

INPUT_DIM = 28
HIDDEN_DIM = 128
OUTPUT_DIM = 10
LEARNING_RATE = 0.005
batch_size = 64
EPOCH = 5
num_train_samples = 5000
num_test_samples = 1000


transform = transforms.ToTensor()
trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_subset = Subset(trainset, range(num_train_samples))
test_subset = Subset(testset, range(num_test_samples))
train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)


def one_hot(label, num_classes=10):
    y = np.zeros((num_classes, 1))
    y[label] = 1
    return y


def train_custom(model_type='rnn'):
    if model_type == 'rnn':
        model = RNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    else:
        model = LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

    model_name = model_type.upper()
    print(f"\nTraining custom {model_name}...")

    start_time = time.time()
    for epoch in range(EPOCH):
        correct = 0
        for x, label in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}"):
            x = x.squeeze().numpy()
            label = label.item()

            output, _ = model.forward(x)
            pred = np.argmax(output)
            if pred == label:
                correct += 1

            target = one_hot(label, OUTPUT_DIM)
            dLdy = 2 * (output - target)
            model.backward(dLdy, LEARNING_RATE)

        acc = correct / num_train_samples * 100
        print(f"Epoch {epoch+1} Accuracy: {acc:.2f}%")
    total_time = time.time() - start_time
    return model, acc, total_time


class TorchRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=28, hidden_size=HIDDEN_DIM, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


class TorchLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def train_torch(model, model_name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining PyTorch {model_name}...")
    start_time = time.time()

    for epoch in range(EPOCH):
        correct = 0
        for x, label in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}"):
            x = x.to(DEVICE)
            x = x.squeeze(1)
            label = label.to(DEVICE)

            output = model(x)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            pred = output.argmax(dim=1)
            if pred.item() == label.item():
                correct += 1

        acc = correct / num_train_samples * 100
        print(f"Epoch {epoch+1} Accuracy: {acc:.2f}%")

    total_time = time.time() - start_time
    return model, acc, total_time


if __name__ == "__main__":
    rnn_model, rnn_acc, rnn_time = train_custom('rnn')
    lstm_model, lstm_acc, lstm_time = train_custom('lstm')

    torch_rnn_model, trnn_acc, trnn_time = train_torch(TorchRNN(), "RNN")
    torch_lstm_model, tlstm_acc, tlstm_time = train_torch(TorchLSTM(), "LSTM")

    # Summary
    print("\nFinal Comparison:")
    print(f"Custom RNN       - Acc: {rnn_acc:.2f}%, Time: {rnn_time:.2f}s")
    print(f"PyTorch RNN      - Acc: {trnn_acc:.2f}%, Time: {trnn_time:.2f}s")
    print(f"Custom LSTM      - Acc: {lstm_acc:.2f}%, Time: {lstm_time:.2f}s")
    print(f"PyTorch LSTM     - Acc: {tlstm_acc:.2f}%, Time: {tlstm_time:.2f}s")
