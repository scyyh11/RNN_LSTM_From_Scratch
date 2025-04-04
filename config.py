import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Model architecture
INPUT_DIM = 28
HIDDEN_DIM = 128
OUTPUT_DIM = 10

# Training configuration
LEARNING_RATE = 0.005
BATCH_SIZE = 64
EPOCHS = 10

# Dataset sizes
NUM_TRAIN_SAMPLES = 5000
NUM_TEST_SAMPLES = 1000
