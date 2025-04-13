# 文件：models/neural_net.py
# 文件：models/neural_net.py

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', reg_lambda=0.0):
        self.params = {}
        self.reg_lambda = reg_lambda
        self.activation = activation

        # 权重初始化：He 初始化法对 ReLU 效果较好
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.params['b1'] = np.zeros((1, hidden_size))
        self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.params['b2'] = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def compute_loss(self, X, y):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']

        # Forward
        z1 = X @ W1 + b1
        a1 = self.relu(z1)
        z2 = a1 @ W2 + b2
        probs = self.softmax(z2)

        # Cross-entropy loss
        N = X.shape[0]
        correct_logprobs = -np.log(probs[range(N), y] + 1e-8)
        data_loss = np.sum(correct_logprobs) / N

        # L2 regularization
        reg_loss = 0.5 * self.reg_lambda * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        loss = data_loss + reg_loss

        self.cache = (X, z1, a1, z2, probs, y)
        return loss, probs

    def backward(self, X, y, probs):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        X, z1, a1, z2, probs, y = self.cache
        N = X.shape[0]

        # Gradient on scores
        dscores = probs.copy()
        dscores[range(N), y] -= 1
        dscores /= N

        # Backpropagation
        dW2 = a1.T @ dscores + self.reg_lambda * W2
        db2 = np.sum(dscores, axis=0, keepdims=True)

        da1 = dscores @ W2.T
        dz1 = da1 * self.relu_derivative(z1)

        dW1 = X.T @ dz1 + self.reg_lambda * W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return grads

    def update_params(self, grads, learning_rate):
        for param in self.params:
            self.params[param] -= learning_rate * grads[param]

    def predict(self, X):
        z1 = X @ self.params['W1'] + self.params['b1']
        a1 = self.relu(z1)
        z2 = a1 @ self.params['W2'] + self.params['b2']
        probs = self.softmax(z2)
        return np.argmax(probs, axis=1)
