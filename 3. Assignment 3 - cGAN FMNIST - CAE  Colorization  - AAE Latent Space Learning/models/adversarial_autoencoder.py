import numpy as np
from models.layers import Conv2D, MaxPool2D, Upsample2D, BatchNorm2D, Dense, Flatten, Reshape
from models.activations import ReLU, LeakyReLU, Sigmoid


class AdversarialAutoencoder:

    def __init__(self, latent_dim=64):
        self.latent_dim = latent_dim

        # ---- Encoder (Conv + Dense to latent) ----
        self.enc_conv1 = Conv2D(3, 16, kernel_size=3, stride=1, pad=1)
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

        self.enc_flatten = Flatten()
        self.enc_fc = Dense(4096, latent_dim)

        # ---- Decoder (Dense from latent + Conv) ----
        self.dec_fc = Dense(latent_dim, 4096)
        self.dec_fc_act = ReLU()
        self.dec_reshape = Reshape((64, 8, 8))

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

        # ---- Discriminator (MLP on latent z) ----
        self.disc_fc1 = Dense(latent_dim, 128)
        self.disc_act1 = LeakyReLU(alpha=0.2)
        self.disc_fc2 = Dense(128, 64)
        self.disc_act2 = LeakyReLU(alpha=0.2)
        self.disc_fc3 = Dense(64, 1)
        self.disc_act3 = Sigmoid()

        self.encoder_layers = [
            self.enc_conv1, self.enc_bn1,
            self.enc_conv2, self.enc_bn2,
            self.enc_conv3, self.enc_bn3,
            self.enc_fc,
        ]
        self.decoder_layers = [
            self.dec_fc,
            self.dec_conv1, self.dec_bn1,
            self.dec_conv2, self.dec_bn2,
            self.dec_conv3,
        ]
        self.discriminator_layers = [
            self.disc_fc1, self.disc_fc2, self.disc_fc3,
        ]

    # ---------------------------------------------------------
    # Encoder
    # ---------------------------------------------------------
    def encode(self, x):
        x = self.enc_pool1.forward(self.enc_act1.forward(
            self.enc_bn1.forward(self.enc_conv1.forward(x))))
        x = self.enc_pool2.forward(self.enc_act2.forward(
            self.enc_bn2.forward(self.enc_conv2.forward(x))))
        x = self.enc_pool3.forward(self.enc_act3.forward(
            self.enc_bn3.forward(self.enc_conv3.forward(x))))
        x = self.enc_flatten.forward(x)
        z = self.enc_fc.forward(x)
        return z

    def encode_backward(self, dz):
        dout = self.enc_fc.backward(dz)
        dout = self.enc_flatten.backward(dout)

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

    # ---------------------------------------------------------
    # Decoder
    # ---------------------------------------------------------
    def decode(self, z):
        x = self.dec_fc_act.forward(self.dec_fc.forward(z))
        x = self.dec_reshape.forward(x)

        x = self.dec_up1.forward(x)
        x = self.dec_act1.forward(
            self.dec_bn1.forward(self.dec_conv1.forward(x)))

        x = self.dec_up2.forward(x)
        x = self.dec_act2.forward(
            self.dec_bn2.forward(self.dec_conv2.forward(x)))

        x = self.dec_up3.forward(x)
        x = self.dec_act3.forward(self.dec_conv3.forward(x))

        return x

    def decode_backward(self, dout):
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

        dout = self.dec_reshape.backward(dout)
        dout = self.dec_fc_act.backward(dout)
        dz = self.dec_fc.backward(dout)
        return dz

    # ---------------------------------------------------------
    # Discriminator
    # ---------------------------------------------------------
    def discriminate(self, z):
        x = self.disc_act1.forward(self.disc_fc1.forward(z))
        x = self.disc_act2.forward(self.disc_fc2.forward(x))
        x = self.disc_act3.forward(self.disc_fc3.forward(x))
        return x

    def discriminate_backward(self, dout):
        dout = self.disc_act3.backward(dout)
        dout = self.disc_fc3.backward(dout)
        dout = self.disc_act2.backward(dout)
        dout = self.disc_fc2.backward(dout)
        dout = self.disc_act1.backward(dout)
        dz = self.disc_fc1.backward(dout)
        return dz

    # ---------------------------------------------------------
    # Full forward (encode + decode)
    # ---------------------------------------------------------
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    # ---------------------------------------------------------
    # Training mode
    # ---------------------------------------------------------
    def set_training(self, mode=True):
        for bn in [self.enc_bn1, self.enc_bn2, self.enc_bn3,
                   self.dec_bn1, self.dec_bn2]:
            bn.training = mode

    def get_encoder_layers(self):
        return self.encoder_layers

    def get_decoder_layers(self):
        return self.decoder_layers

    def get_discriminator_layers(self):
        return self.discriminator_layers

    def get_autoencoder_layers(self):
        return self.encoder_layers + self.decoder_layers
