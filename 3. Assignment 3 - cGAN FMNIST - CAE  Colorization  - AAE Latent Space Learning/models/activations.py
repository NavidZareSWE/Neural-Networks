import numpy as np


class ReLU:

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0).astype(np.float64)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

    def params_and_grads(self):
        return []


class LeakyReLU:

    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0).astype(np.float64)
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, dout):
        return dout * np.where(self.mask > 0, 1.0, self.alpha)

    def params_and_grads(self):
        return []


class Sigmoid:

    def __init__(self):
        self.out = None

    def forward(self, x):
        x = np.clip(x, -500, 500)
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * self.out * (1.0 - self.out)

    def params_and_grads(self):
        return []


class Tanh:

    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out ** 2)

    def params_and_grads(self):
        return []
