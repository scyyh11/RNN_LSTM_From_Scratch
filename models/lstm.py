import numpy as np
from .activations import tanh, sigmoid


class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.cache = []

        scale = 0.1

        # Initialize Input Gate
        self.Wi = np.random.randn(self.hidden_dim, self.input_dim) * scale
        self.Ui = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.bi = np.zeros((self.hidden_dim, 1))

        # Initialize Forget Gate
        self.Wf = np.random.randn(self.hidden_dim, self.input_dim) * scale
        self.Uf = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.bf = np.zeros((self.hidden_dim, 1))

        # Initialize Output Gate
        self.Wo = np.random.randn(self.hidden_dim, self.input_dim) * scale
        self.Uo = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.bo = np.zeros((self.hidden_dim, 1))

        # Initialize Cell
        self.Wc = np.random.randn(self.hidden_dim, self.input_dim) * scale
        self.Uc = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.bc = np.zeros((self.hidden_dim, 1))

        # Initialize Output Layer
        self.Wy = np.random.randn(self.output_dim, self.hidden_dim) * scale
        self.by = np.zeros((self.output_dim, 1))

    def forward(self, x):
        """
            Performs the forward pass of the LSTM over a sequence of inputs.

            At each time step, the LSTM updates:
            - The input gate (`i_t`) which controls how much new information to store.
            - The forget gate (`f_t`) which determines how much of the previous memory to keep.
            - The cell candidate (`c_tilde`) which proposes new content to add to the memory.
            - The output gate (`o_t`) which controls how much of the cell state to output.

            These gates work together to update the hidden state (`h_t`) and the cell state (`c_t`).
            After the sequence is fully processed, the final hidden state is used to compute the output.

            Args:
                X (np.ndarray): Input sequence of shape (sequence_length, input_dim),
                                where each row represents the input at one time step.

            Returns:
                y (np.ndarray): Output computed from the final hidden state, shape (output_dim, 1).
                h_t (np.ndarray): Final hidden state of the LSTM, shape (hidden_dim, 1).
        """
        h_prev = np.zeros((self.hidden_dim, 1))
        c_prev = np.zeros((self.hidden_dim, 1))

        # Clear cache
        self.cache = []

        # Process every time step
        for t in range(x.shape[0]):
            # Reshape the input row into a column
            x_t = x[t].reshape(-1, 1)

            i_t = sigmoid(self.Wi @ x_t + self.Ui @ h_prev + self.bi)
            f_t = sigmoid(self.Wf @ x_t + self.Uf @ h_prev + self.bf)
            o_t = sigmoid(self.Wo @ x_t + self.Uo @ h_prev + self.bo)
            c_tilde = tanh(self.Wc @ x_t + self.Uc @ h_prev + self.bc)
            c_t = f_t * c_prev + i_t * c_tilde
            h_t = o_t * tanh(c_t)

            self.cache.append((x_t, h_prev, c_prev, i_t, f_t, o_t, c_tilde, c_t, h_t))

            h_prev = h_t
            c_prev = c_t

        y = np.dot(self.Wy ,h_t) + self.by
        return y, h_t

    def backward(self, dLdy, learning_rate=0.001, clip_value=5.0):
        """
            Performs Backpropagation Through Time (BPTT) for the LSTM.

            Compute gradients of the loss with respect to all trainable parameters
            of the LSTM, using the chain rule over the unrolled computation graph.
            It also applies gradient clipping and parameter updates using simple
            gradient descent.

            Args:
                dLdy (np.ndarray): Gradient of the loss with respect to the final output y,
                                   shape should match output bias (output_dim, 1).
                learning_rate (float): Learning rate for parameter update.
                clip_value (float): Threshold for gradient clipping to prevent exploding gradients.

            Returns:
                None
            """
        dWi, dUi, dbi = np.zeros_like(self.Wi), np.zeros_like(self.Ui), np.zeros_like(self.bi)
        dWf, dUf, dbf = np.zeros_like(self.Wf), np.zeros_like(self.Uf), np.zeros_like(self.bf)
        dWo, dUo, dbo = np.zeros_like(self.Wo), np.zeros_like(self.Uo), np.zeros_like(self.bo)
        dWc, dUc, dbc = np.zeros_like(self.Wc), np.zeros_like(self.Uc), np.zeros_like(self.bc)
        dWy, dby = np.zeros_like(self.Wy), np.zeros_like(self.by)

        dLdy = dLdy.reshape(self.by.shape)

        h_last = self.cache[-1][8]
        dWy += dLdy @ h_last.T
        dby += dLdy

        dh_next = self.Wy.T @ dLdy
        dc_next = np.zeros_like(dh_next)

        for t in reversed(range(len(self.cache))):
            (x_t, h_prev, c_prev, i_t, f_t, o_t, c_tilde, c_t, h_t) = self.cache[t]

            do = dh_next * tanh(c_t) * o_t * (1 - o_t)
            dc = dh_next * o_t * (1 - tanh(c_t) ** 2) + dc_next
            di = dc * c_tilde * i_t * (1 - i_t)
            df = dc * c_prev * f_t * (1 - f_t)
            dc_tilde = dc * i_t * (1 - c_tilde ** 2)

            dWi += np.dot(di, x_t.T)
            dUi += np.dot(di, h_prev.T)
            dbi += di

            dWf += np.dot(df, x_t.T)
            dUf += np.dot(df, h_prev.T)
            dbf += df

            dWo += np.dot(do, x_t.T)
            dUo += np.dot(do, h_prev.T)
            dbo += do

            dWc += np.dot(dc_tilde, x_t.T)
            dUc += np.dot(dc_tilde, h_prev.T)
            dbc += dc_tilde

            dh_prev = (
                    np.dot(self.Ui.T, di) +
                    np.dot(self.Uf.T, df) +
                    np.dot(self.Uo.T, do) +
                    np.dot(self.Uc.T, dc_tilde)
            )

            dc_prev = dc * f_t

            dh_next = dh_prev
            dc_next = dc_prev

        for grad in [dWi, dUi, dbi, dWf, dUf, dbf, dWo, dUo, dbo, dWc, dUc, dbc, dWy, dby]:
            np.clip(grad, -clip_value, clip_value, out=grad)

        self.Wi -= learning_rate * dWi
        self.Ui -= learning_rate * dUi
        self.bi -= learning_rate * dbi

        self.Wf -= learning_rate * dWf
        self.Uf -= learning_rate * dUf
        self.bf -= learning_rate * dbf

        self.Wo -= learning_rate * dWo
        self.Uo -= learning_rate * dUo
        self.bo -= learning_rate * dbo

        self.Wc -= learning_rate * dWc
        self.Uc -= learning_rate * dUc
        self.bc -= learning_rate * dbc

        self.Wy -= learning_rate * dWy
        self.by -= learning_rate * dby
