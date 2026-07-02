import numpy as np


class BCELoss:

    def __init__(self, eps=1e-7):
        self.eps = eps
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        self.pred = np.clip(pred, self.eps, 1 - self.eps)
        self.target = target
        loss = -np.mean(target * np.log(self.pred) + (1 - target) * np.log(1 - self.pred))
        return loss

    def backward(self):
        N = self.pred.shape[0]
        grad = -(self.target / self.pred - (1 - self.target) / (1 - self.pred)) / N
        return grad


class MSELoss:

    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        loss = np.mean((pred - target) ** 2)
        return loss

    def backward(self):
        N = self.pred.size
        grad = 2.0 * (self.pred - self.target) / N
        return grad
