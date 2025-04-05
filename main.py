from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models import torch_rnn, torch_lstm
from models.rnn import RNN
from models.lstm import LSTM
from config import INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES, BATCH_SIZE
from utils import helpers

transform = transforms.ToTensor()
trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_subset = Subset(trainset, range(NUM_TRAIN_SAMPLES))
test_subset = Subset(testset, range(NUM_TEST_SAMPLES))

# Use batch_size = 1 for custom models, regular for torch models
train_loader_custom = DataLoader(train_subset, batch_size=1, shuffle=True)
test_loader_custom = DataLoader(test_subset, batch_size=1, shuffle=False)
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

if __name__ == "__main__":
    # Custom
    rnn_model, rnn_acc = helpers.train(RNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM),
                                       train_loader_custom, "Custom RNN", is_custom=True)
    lstm_model, lstm_acc = helpers.train(LSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM),
                                         train_loader_custom, "Custom LSTM", is_custom=True)

    # PyTorch
    torch_rnn_model, trnn_acc = helpers.train(torch_rnn.TorchRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM),
                                              train_loader, "PyTorch RNN")
    torch_lstm_model, tlstm_acc = helpers.train(torch_lstm.TorchLSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM),
                                                train_loader, "PyTorch LSTM")

    print("\nFinal Comparison:")
    print(f"Custom RNN       - Train Acc: {rnn_acc:.2f}%")
    print(f"PyTorch RNN      - Train Acc: {trnn_acc:.2f}%")
    print(f"Custom LSTM      - Train Acc: {lstm_acc:.2f}%")
    print(f"PyTorch LSTM     - Train Acc: {tlstm_acc:.2f}%")

    print("\nEvaluating on test set...")
    helpers.evaluate(rnn_model, test_loader_custom, "Custom RNN", is_custom=True)
    helpers.evaluate(lstm_model, test_loader_custom, "Custom LSTM", is_custom=True)
    helpers.evaluate(torch_rnn_model, test_loader, "PyTorch RNN")
    helpers.evaluate(torch_lstm_model, test_loader, "PyTorch LSTM")
