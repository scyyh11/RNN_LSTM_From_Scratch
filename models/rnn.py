import numpy as np
from .activations import tanh


class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.cache = []

        # Initialize Weight (scale factor = 0,1)
        scale = 0.1
        self.Wx = np.random.randn(self.hidden_dim, self.input_dim) * scale
        self.Wh = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.Wy = np.random.randn(self.output_dim, self.hidden_dim) * scale

        # Initialize Bias
        self.bh = np.zeros((self.hidden_dim, 1))
        self.by = np.zeros((self.output_dim, 1))

    def forward(self, x):
        """
        Forward pass through the RNN for one input sequence.

        At each time step, the RNN updates its hidden state using the current input
        and the previous hidden state. After the full sequence is processed, the final
        hidden state is used to compute the output.

        Args:
            x (np.ndarray): Input sequence of shape (sequence_length, input_dim).
                            Each row represents one time step.

        Returns:
            y (np.ndarray): Final output of the network, shape (output_dim, 1).
            h_t (np.ndarray): Final hidden state of the RNN, shape (hidden_dim, 1).
        """
        # Initialize hidden state
        h_t = np.zeros((self.hidden_dim, 1))

        # Clear cache
        self.cache = []

        # Process every time step
        for t in range(x.shape[0]):
            x_t = x[t].reshape(-1, 1)
            h_prev = h_t.copy()
            h_t = tanh(np.dot(self.Wh, h_t) + np.dot(self.Wx, x_t) + self.bh)
            self.cache.append((h_prev, x_t.copy(), h_t.copy()))

        y = np.dot(self.Wy, h_t) + self.by

        return y, h_t

    def backward(self, dLdy, learning_rate=0.001, clip_value=5.0):
        """
        Performs Backpropagation Through Time (BPTT) for the RNN.

        Compute gradients of the loss with respect to the model
        parameters (Wx, Wh, Wy, bh, by) using the chain rule, and updates the
        parameters using gradient descent.

        Args:
            dLdy (np.ndarray): Gradient of the loss with respect to the final output y.
            learning_rate (float): Step size for updating weights and biases.
            clip_value (float): Threshold for gradient clipping to avoid exploding gradients.
        Returns:
            None
        """
        # Initialize partial derivatives
        dWh = np.zeros_like(self.Wh)
        dWx = np.zeros_like(self.Wx)
        dWy = np.zeros_like(self.Wy)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dLdy = dLdy.reshape(self.by.shape)

        dby = dLdy
        dWy = np.dot(dLdy, self.cache[-1][2].T)
        dh = np.dot(self.Wy.T, dLdy)

        for t in reversed(range(len(self.cache))):
            h_prev, x_t, h_t = self.cache[t]

            dtanh = dh * (1 - h_t ** 2)
            dWh += np.dot(dtanh, h_prev.T)
            dWx += np.dot(dtanh, x_t.T)
            dbh += dtanh

            if t>0:
                dh = np.dot(self.Wh.T, dtanh)
            else:
                dh = np.zeros_like(h_t)

        # Gradients clipping
        for grad in [dWh, dWx, dWy, dbh, dby]:
            np.clip(grad, -clip_value, clip_value, out=grad)

        # Update parameters
        self.Wh -= learning_rate * dWh
        self.Wx -= learning_rate * dWx
        self.Wy -= learning_rate * dWy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby
