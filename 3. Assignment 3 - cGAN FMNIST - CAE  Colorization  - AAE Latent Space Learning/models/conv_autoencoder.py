import numpy as np
from models.layers import Conv2D, MaxPool2D, Upsample2D, BatchNorm2D
from models.activations import ReLU, Sigmoid


class ConvAutoencoder:

    def __init__(self):
        # ---- Encoder ----
        self.enc_conv1 = Conv2D(1, 16, kernel_size=3, stride=1, pad=1)
        self.enc_bn1 = BatchNorm2D(16)
        self.enc_act1 = ReLU()
        self.enc_pool1 = MaxPool2D(pool_size=2, stride=2)

        self.enc_conv2 = Conv2D(16, 32, kernel_size=3, stride=1, pad=1)
        self.enc_bn2 = BatchNorm2D(32)
        self.enc_act2 = ReLU()
        self.enc_pool2 = MaxPool2D(pool_size=2, stride=2)

        self.enc_conv3 = Conv2D(32, 64, kernel_size=3, stride=1, pad=1)
        self.enc_bn3 = BatchNorm2D(64)
        self.enc_act3 = ReLU()
        self.enc_pool3 = MaxPool2D(pool_size=2, stride=2)

        # ---- Decoder ----
        self.dec_up1 = Upsample2D(scale_factor=2)
        self.dec_conv1 = Conv2D(64, 32, kernel_size=3, stride=1, pad=1)
        self.dec_bn1 = BatchNorm2D(32)
        self.dec_act1 = ReLU()

        self.dec_up2 = Upsample2D(scale_factor=2)
        self.dec_conv2 = Conv2D(32, 16, kernel_size=3, stride=1, pad=1)
        self.dec_bn2 = BatchNorm2D(16)
        self.dec_act2 = ReLU()

        self.dec_up3 = Upsample2D(scale_factor=2)
        self.dec_conv3 = Conv2D(16, 3, kernel_size=3, stride=1, pad=1)
        self.dec_act3 = Sigmoid()

        self.layers_list = [
            self.enc_conv1, self.enc_bn1,
            self.enc_conv2, self.enc_bn2,
            self.enc_conv3, self.enc_bn3,
            self.dec_conv1, self.dec_bn1,
            self.dec_conv2, self.dec_bn2,
            self.dec_conv3,
        ]

    def forward(self, x):
        x = self.enc_conv1.forward(x)
        x = self.enc_bn1.forward(x)
        x = self.enc_act1.forward(x)
        x = self.enc_pool1.forward(x)

        x = self.enc_conv2.forward(x)
        x = self.enc_bn2.forward(x)
        x = self.enc_act2.forward(x)
        x = self.enc_pool2.forward(x)

        x = self.enc_conv3.forward(x)
        x = self.enc_bn3.forward(x)
        x = self.enc_act3.forward(x)
        x = self.enc_pool3.forward(x)

        x = self.dec_up1.forward(x)
        x = self.dec_conv1.forward(x)
        x = self.dec_bn1.forward(x)
        x = self.dec_act1.forward(x)

        x = self.dec_up2.forward(x)
        x = self.dec_conv2.forward(x)
        x = self.dec_bn2.forward(x)
        x = self.dec_act2.forward(x)

        x = self.dec_up3.forward(x)
        x = self.dec_conv3.forward(x)
        x = self.dec_act3.forward(x)

        return x

    def backward(self, dout):
        dout = self.dec_act3.backward(dout)
        dout = self.dec_conv3.backward(dout)
        dout = self.dec_up3.backward(dout)

        dout = self.dec_act2.backward(dout)
        dout = self.dec_bn2.backward(dout)
        dout = self.dec_conv2.backward(dout)
        dout = self.dec_up2.backward(dout)

        dout = self.dec_act1.backward(dout)
        dout = self.dec_bn1.backward(dout)
        dout = self.dec_conv1.backward(dout)
        dout = self.dec_up1.backward(dout)

        dout = self.enc_pool3.backward(dout)
        dout = self.enc_act3.backward(dout)
        dout = self.enc_bn3.backward(dout)
        dout = self.enc_conv3.backward(dout)

        dout = self.enc_pool2.backward(dout)
        dout = self.enc_act2.backward(dout)
        dout = self.enc_bn2.backward(dout)
        dout = self.enc_conv2.backward(dout)

        dout = self.enc_pool1.backward(dout)
        dout = self.enc_act1.backward(dout)
        dout = self.enc_bn1.backward(dout)
        dout = self.enc_conv1.backward(dout)

    def set_training(self, mode=True):
        for bn in [self.enc_bn1, self.enc_bn2, self.enc_bn3,
                    self.dec_bn1, self.dec_bn2]:
            bn.training = mode

    def get_layers(self):
        return self.layers_list
