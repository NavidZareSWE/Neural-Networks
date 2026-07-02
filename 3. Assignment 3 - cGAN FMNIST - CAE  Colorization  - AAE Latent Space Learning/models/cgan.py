import numpy as np
from models.layers import Dense, BatchNorm1D, Dropout
from models.activations import LeakyReLU, Sigmoid, Tanh


def one_hot(labels, num_classes):
    N = labels.shape[0]
    oh = np.zeros((N, num_classes))
    oh[np.arange(N), labels.astype(int)] = 1.0
    return oh


class Generator:

    def __init__(self, latent_dim=100, num_classes=10):
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        input_dim = latent_dim + num_classes

        self.fc1 = Dense(input_dim, 256)
        self.bn1 = BatchNorm1D(256)
        self.act1 = LeakyReLU(0.2)

        self.fc2 = Dense(256, 512)
        self.bn2 = BatchNorm1D(512)
        self.act2 = LeakyReLU(0.2)

        self.fc3 = Dense(512, 1024)
        self.bn3 = BatchNorm1D(1024)
        self.act3 = LeakyReLU(0.2)

        self.fc4 = Dense(1024, 784)
        self.act_out = Tanh()

        self.layers_list = [
            self.fc1, self.bn1,
            self.fc2, self.bn2,
            self.fc3, self.bn3,
            self.fc4,
        ]

    def forward(self, z, labels):
        label_oh = one_hot(labels, self.num_classes)
        x = np.concatenate([z, label_oh], axis=1)

        x = self.fc1.forward(x)
        x = self.bn1.forward(x)
        x = self.act1.forward(x)

        x = self.fc2.forward(x)
        x = self.bn2.forward(x)
        x = self.act2.forward(x)

        x = self.fc3.forward(x)
        x = self.bn3.forward(x)
        x = self.act3.forward(x)

        x = self.fc4.forward(x)
        x = self.act_out.forward(x)
        return x

    def backward(self, dout):
        dout = self.act_out.backward(dout)
        dout = self.fc4.backward(dout)

        dout = self.act3.backward(dout)
        dout = self.bn3.backward(dout)
        dout = self.fc3.backward(dout)

        dout = self.act2.backward(dout)
        dout = self.bn2.backward(dout)
        dout = self.fc2.backward(dout)

        dout = self.act1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.fc1.backward(dout)

    def set_training(self, mode=True):
        self.bn1.training = mode
        self.bn2.training = mode
        self.bn3.training = mode

    def get_layers(self):
        return self.layers_list


class Discriminator:

    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.image_dim = 784

        input_dim = 784 + num_classes

        self.fc1 = Dense(input_dim, 512)
        self.act1 = LeakyReLU(0.2)
        self.drop1 = Dropout(0.3)

        self.fc2 = Dense(512, 256)
        self.act2 = LeakyReLU(0.2)
        self.drop2 = Dropout(0.3)

        self.fc3 = Dense(256, 1)
        self.act_out = Sigmoid()

        self.layers_list = [
            self.fc1, self.drop1,
            self.fc2, self.drop2,
            self.fc3,
        ]

    def forward(self, images, labels):
        label_oh = one_hot(labels, self.num_classes)
        x = np.concatenate([images, label_oh], axis=1)

        x = self.fc1.forward(x)
        x = self.act1.forward(x)
        x = self.drop1.forward(x)

        x = self.fc2.forward(x)
        x = self.act2.forward(x)
        x = self.drop2.forward(x)

        x = self.fc3.forward(x)
        x = self.act_out.forward(x)
        return x

    def backward(self, dout):
        dout = self.act_out.backward(dout)
        dout = self.fc3.backward(dout)

        dout = self.drop2.backward(dout)
        dout = self.act2.backward(dout)
        dout = self.fc2.backward(dout)

        dout = self.drop1.backward(dout)
        dout = self.act1.backward(dout)
        dout = self.fc1.backward(dout)

        return dout[:, :self.image_dim]

    def set_training(self, mode=True):
        self.drop1.training = mode
        self.drop2.training = mode

    def get_layers(self):
        return self.layers_list
