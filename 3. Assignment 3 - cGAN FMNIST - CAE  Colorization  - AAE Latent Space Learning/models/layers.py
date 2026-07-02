import numpy as np


# ---------------------------------------------------------
# Utility: im2col / col2im for efficient convolution
# ---------------------------------------------------------

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0),
                 (pad, pad), (pad, pad)], mode='constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h,
                      filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


# ---------------------------------------------------------
# Dense (Fully Connected) Layer
# ---------------------------------------------------------

class Dense:

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.randn(
            in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        self.dW = None
        self.db = None
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        return dout @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW, 'W'), (self.b, self.db, 'b')]


# ---------------------------------------------------------
# Conv2D Layer
# ---------------------------------------------------------

class Conv2D:

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        fan_in = in_channels * kernel_size * kernel_size
        self.W = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(out_channels)

        self.dW = None
        self.db = None
        self.x = None
        self.col = None
        self.col_W = None

    def forward(self, x):
        N, C, H, W_in = x.shape
        out_h = (H + 2 * self.pad - self.kernel_size) // self.stride + 1
        out_w = (W_in + 2 * self.pad - self.kernel_size) // self.stride + 1

        col = im2col(x, self.kernel_size, self.kernel_size,
                     self.stride, self.pad)
        col_W = self.W.reshape(self.out_channels, -1).T

        out = col @ col_W + self.b
        out = out.reshape(
            N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        N, C, H, W_in = self.x.shape
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        self.db = np.sum(dout_flat, axis=0)
        self.dW = (self.col.T @ dout_flat).T.reshape(self.W.shape)

        dcol = dout_flat @ self.col_W.T
        dx = col2im(dcol, self.x.shape, self.kernel_size,
                    self.kernel_size, self.stride, self.pad)
        return dx

    def params_and_grads(self):
        return [(self.W, self.dW, 'W'), (self.b, self.db, 'b')]


# ---------------------------------------------------------
# MaxPool2D Layer
# ---------------------------------------------------------

class MaxPool2D:

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        col = im2col(x, self.pool_size, self.pool_size, self.stride, 0)
        col = col.reshape(-1, self.pool_size * self.pool_size)

        self.arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        return out

    def backward(self, dout):
        N, C, H, W = self.x.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        dout_flat = dout.transpose(0, 2, 3, 1).flatten()
        pool_area = self.pool_size * self.pool_size

        dmax = np.zeros((dout_flat.size, pool_area))
        dmax[np.arange(dout_flat.size), self.arg_max] = dout_flat

        dcol = dmax.reshape(N, out_h, out_w, C, self.pool_size, self.pool_size)
        dcol = dcol.reshape(N * out_h * out_w, -1)

        dx = col2im(dcol, self.x.shape, self.pool_size,
                    self.pool_size, self.stride, 0)
        return dx

    def params_and_grads(self):
        return []


# ---------------------------------------------------------
# Upsample2D Layer (Nearest Neighbor)
# ---------------------------------------------------------

class Upsample2D:

    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        s = self.scale_factor
        return x.repeat(s, axis=2).repeat(s, axis=3)

    def backward(self, dout):
        N, C, H, W = self.x_shape
        s = self.scale_factor
        dout_reshaped = dout.reshape(N, C, H, s, W, s)
        return dout_reshaped.sum(axis=(3, 5))

    def params_and_grads(self):
        return []


# ---------------------------------------------------------
# Batch Normalization
# ---------------------------------------------------------

class BatchNorm1D:

    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.momentum = momentum
        self.eps = eps

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.dgamma = None
        self.dbeta = None
        self.x_norm = None
        self.std_inv = None
        self.x_centered = None
        self.training = True

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = self.momentum * \
                self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * \
                self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        self.std_inv = 1.0 / np.sqrt(var + self.eps)
        self.x_centered = x - mean
        self.x_norm = self.x_centered * self.std_inv
        return self.gamma * self.x_norm + self.beta

    def backward(self, dout):
        N = dout.shape[0]
        self.dgamma = np.sum(dout * self.x_norm, axis=0)
        self.dbeta = np.sum(dout, axis=0)

        dx_norm = dout * self.gamma
        dx = (1.0 / N) * self.std_inv * (
            N * dx_norm - np.sum(dx_norm, axis=0) -
            self.x_norm * np.sum(dx_norm * self.x_norm, axis=0)
        )
        return dx

    def params_and_grads(self):
        return [(self.gamma, self.dgamma, 'gamma'), (self.beta, self.dbeta, 'beta')]


class BatchNorm2D:

    def __init__(self, num_channels, momentum=0.9, eps=1e-5):
        self.num_channels = num_channels
        self.gamma = np.ones(num_channels)
        self.beta = np.zeros(num_channels)
        self.momentum = momentum
        self.eps = eps

        self.running_mean = np.zeros(num_channels)
        self.running_var = np.ones(num_channels)

        self.dgamma = None
        self.dbeta = None
        self.x_norm = None
        self.std_inv = None
        self.x_centered = None
        self.training = True

    def forward(self, x):
        N, C, H, W = x.shape
        if self.training:
            mean = x.mean(axis=(0, 2, 3))
            var = x.var(axis=(0, 2, 3))
            self.running_mean = self.momentum * \
                self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * \
                self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        self.std_inv = 1.0 / np.sqrt(var + self.eps)
        mean_r = mean.reshape(1, C, 1, 1)
        std_inv_r = self.std_inv.reshape(1, C, 1, 1)

        self.x_centered = x - mean_r
        self.x_norm = self.x_centered * std_inv_r

        gamma_r = self.gamma.reshape(1, C, 1, 1)
        beta_r = self.beta.reshape(1, C, 1, 1)
        return gamma_r * self.x_norm + beta_r

    def backward(self, dout):
        N, C, H, W = dout.shape
        M = N * H * W

        gamma_r = self.gamma.reshape(1, C, 1, 1)
        self.dgamma = np.sum(dout * self.x_norm, axis=(0, 2, 3))
        self.dbeta = np.sum(dout, axis=(0, 2, 3))

        dx_norm = dout * gamma_r
        std_inv_r = self.std_inv.reshape(1, C, 1, 1)

        dx = (1.0 / M) * std_inv_r * (
            M * dx_norm -
            np.sum(dx_norm, axis=(0, 2, 3), keepdims=True) -
            self.x_norm * np.sum(dx_norm * self.x_norm,
                                 axis=(0, 2, 3), keepdims=True)
        )
        return dx

    def params_and_grads(self):
        return [(self.gamma, self.dgamma, 'gamma'), (self.beta, self.dbeta, 'beta')]


# ---------------------------------------------------------
# Dropout Mechanism
# ---------------------------------------------------------

class Dropout:

    def __init__(self, drop_rate=0.3):
        self.drop_rate = drop_rate
        self.mask = None
        self.training = True

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) >
                         self.drop_rate).astype(np.float64)
            return x * self.mask / (1.0 - self.drop_rate)
        return x

    def backward(self, dout):
        return dout * self.mask / (1.0 - self.drop_rate)

    def params_and_grads(self):
        return []


# ---------------------------------------------------------
# Flatten / Reshape Layers
# ---------------------------------------------------------

class Flatten:

    def __init__(self):
        self.original_shape = None

    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.original_shape)

    def params_and_grads(self):
        return []


class Reshape:

    def __init__(self, target_shape):
        self.target_shape = target_shape
        self.original_shape = None

    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], *self.target_shape)

    def backward(self, dout):
        return dout.reshape(self.original_shape)

    def params_and_grads(self):
        return []


# ---------------------------------------------------------
# Embedding Layer (for label conditioning in cGAN)
# ---------------------------------------------------------

class Embedding:

    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.W = np.random.randn(num_embeddings, embedding_dim) * 0.05
        self.dW = None
        self.indices = None

    def forward(self, indices):
        self.indices = indices.astype(int)
        return self.W[self.indices]

    def backward(self, dout):
        self.dW = np.zeros_like(self.W)
        np.add.at(self.dW, self.indices, dout)
        return None

    def params_and_grads(self):
        return [(self.W, self.dW, 'W')]
