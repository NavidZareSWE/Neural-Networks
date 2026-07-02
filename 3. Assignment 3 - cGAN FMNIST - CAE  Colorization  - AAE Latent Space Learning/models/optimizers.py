import numpy as np


class Adam:

    def __init__(self, lr=0.0002, beta1=0.5, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, layers):
        self.t += 1

        for layer in layers:
            for param, grad, name in layer.params_and_grads():
                key = id(param)
                if key not in self.m:
                    self.m[key] = np.zeros_like(param)
                    self.v[key] = np.zeros_like(param)

                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SGD:

    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def step(self, layers):
        for layer in layers:
            for param, grad, name in layer.params_and_grads():
                key = id(param)
                if key not in self.velocity:
                    self.velocity[key] = np.zeros_like(param)

                self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grad
                param += self.velocity[key]
