from torch import nn, optim
import torch
import numpy as np
from tqdm import tqdm

from config import DEVICE, OUTPUT_DIM, LEARNING_RATE, EPOCHS



def one_hot(label, num_classes=OUTPUT_DIM):
    y = np.zeros((num_classes, 1))
    y[label] = 1
    return y


def train(model, dataloader, model_name="Model", is_custom=False):
    if not is_custom:
        model = model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining {model_name}...")

    for epoch in range(EPOCHS):
        correct = 0
        total = 0

        for x, label in tqdm(dataloader, desc=f"[{model_name}] Epoch {epoch+1}"):
            if is_custom:
                x = x.squeeze().numpy()
                label = label.item()

                output, _ = model.forward(x)
                pred = np.argmax(output)
                if pred == label:
                    correct += 1

                target = one_hot(label, OUTPUT_DIM)
                dLdy = 2 * (output - target)
                model.backward(dLdy, LEARNING_RATE)

            else:
                x = x.to(DEVICE).squeeze(1)
                label = label.to(DEVICE)

                output = model(x)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                pred = output.argmax(dim=1)
                correct += (pred == label).sum().item()

            total += label.size(0) if not is_custom else 1

        acc = correct / total * 100
        print(f"Epoch {epoch+1} Accuracy: {acc:.2f}%")

    return model, acc



def evaluate(model, dataloader, model_name="Model", is_custom=False):
    correct = 0
    total = 0

    if not is_custom:
        model.eval()
        model.to(DEVICE)

    with torch.no_grad():
        for x, label in dataloader:
            if is_custom:
                x = x.squeeze().numpy()
                label = label.item()
                output, _ = model.forward(x)
                pred = np.argmax(output)
                if pred == label:
                    correct += 1
                total += 1
            else:
                # Assume batched tensors
                x = x.to(DEVICE).squeeze(1)
                label = label.to(DEVICE)
                output = model(x)
                pred = output.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

    acc = correct / total * 100
    print(f"{model_name} Test Accuracy: {acc:.2f}%")
    return acc

