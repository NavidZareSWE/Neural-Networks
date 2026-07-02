import numpy as np
import time
from models.adversarial_autoencoder import AdversarialAutoencoder
from models.optimizers import Adam
from models.losses import MSELoss, BCELoss


def train_aae(train_rgb, test_rgb, config):
    batch_size = config.get('batch_size', 8)
    num_epochs = config.get('num_epochs', 10)
    lr = config.get('lr', 0.001)
    latent_dim = config.get('latent_dim', 64)

    print("=" * 70)
    print("  ADVERSARIAL AUTOENCODER TRAINING (Latent Space Learning)")
    print("=" * 70)
    print(f"  Training samples: {len(train_rgb)}")
    print(f"  Test samples:     {len(test_rgb)}")
    print(f"  Image size:       {train_rgb.shape[2]}x{train_rgb.shape[3]}")
    print(f"  Latent dim:       {latent_dim}")
    print(f"  Prior:            N(0, I)")
    print(f"  Batch size:       {batch_size}")
    print(f"  Epochs:           {num_epochs}")
    print(f"  Learning rate:    {lr}")

    print("\n  Architecture:")
    print("    Encoder: Conv(3->16)+BN+ReLU+Pool -> Conv(16->32)+BN+ReLU+Pool")
    print("             -> Conv(32->64)+BN+ReLU+Pool -> Flatten -> Dense(4096,64)")
    print("    Decoder: Dense(64,4096)+ReLU -> Reshape(64,8,8)")
    print("             -> Up+Conv(64->32)+BN+ReLU -> Up+Conv(32->16)+BN+ReLU")
    print("             -> Up+Conv(16->3)+Sigmoid")
    print("    Discriminator: Dense(64,128)+LReLU -> Dense(128,64)+LReLU -> Dense(64,1)+Sig")
    print("=" * 70)

    model = AdversarialAutoencoder(latent_dim=latent_dim)

    opt_ae = Adam(lr=lr, beta1=0.9, beta2=0.999)
    opt_disc = Adam(lr=lr, beta1=0.9, beta2=0.999)
    opt_gen = Adam(lr=lr * 0.5, beta1=0.9, beta2=0.999)

    mse_loss = MSELoss()
    bce_loss = BCELoss()

    ae_params = sum(p.size for l in model.get_autoencoder_layers() for p, _, _ in l.params_and_grads())
    disc_params = sum(p.size for l in model.get_discriminator_layers() for p, _, _ in l.params_and_grads())
    print(f"  Autoencoder params:   {ae_params:,}")
    print(f"  Discriminator params: {disc_params:,}")
    print(f"  Total params:         {ae_params + disc_params:,}")

    history = {
        'recon_losses': [],
        'disc_losses': [],
        'gen_losses': [],
        'val_losses': [],
    }

    n_train = len(train_rgb)
    n_batches = max(1, n_train // batch_size)
    total_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.set_training(True)

        perm = np.random.permutation(n_train)
        train_data = train_rgb[perm]

        epoch_recon = 0.0
        epoch_disc = 0.0
        epoch_gen = 0.0
        n_processed = 0

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_train)
            x_batch = train_data[start:end]
            bs = end - start

            # =========================================================
            # Phase 1: Reconstruction (update encoder + decoder)
            # =========================================================
            x_hat, z = model.forward(x_batch)
            r_loss = mse_loss.forward(x_hat, x_batch)

            d_recon = mse_loss.backward()
            dz_recon = model.decode_backward(d_recon)
            model.encode_backward(dz_recon)
            opt_ae.step(model.get_autoencoder_layers())

            # =========================================================
            # Phase 2: Discriminator (update discriminator only)
            # =========================================================
            z_fake = model.encode(x_batch)
            z_real = np.random.randn(bs, model.latent_dim).astype(np.float32)

            d_real = model.discriminate(z_real)
            labels_real = np.ones((bs, 1))
            d_loss_real = bce_loss.forward(d_real, labels_real)
            d_grad_real = bce_loss.backward()
            model.discriminate_backward(d_grad_real)
            opt_disc.step(model.get_discriminator_layers())

            d_fake = model.discriminate(z_fake)
            labels_fake = np.zeros((bs, 1))
            d_loss_fake = bce_loss.forward(d_fake, labels_fake)
            d_grad_fake = bce_loss.backward()
            model.discriminate_backward(d_grad_fake)
            opt_disc.step(model.get_discriminator_layers())

            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            # =========================================================
            # Phase 3: Generator / Encoder (fool discriminator)
            # =========================================================
            z_fake = model.encode(x_batch)
            d_fake = model.discriminate(z_fake)
            labels_real = np.ones((bs, 1))
            g_loss = bce_loss.forward(d_fake, labels_real)

            d_grad = bce_loss.backward()
            dz_gen = model.discriminate_backward(d_grad)
            model.encode_backward(dz_gen)
            opt_gen.step(model.get_encoder_layers())

            epoch_recon += r_loss * bs
            epoch_disc += d_loss * bs
            epoch_gen += g_loss * bs
            n_processed += bs

        avg_recon = epoch_recon / n_processed
        avg_disc = epoch_disc / n_processed
        avg_gen = epoch_gen / n_processed

        model.set_training(False)
        val_loss = _compute_val_loss(model, test_rgb, mse_loss, batch_size)

        history['recon_losses'].append(avg_recon)
        history['disc_losses'].append(avg_disc)
        history['gen_losses'].append(avg_gen)
        history['val_losses'].append(val_loss)

        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - total_start

        print(f"  Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Recon: {avg_recon:.6f} | Disc: {avg_disc:.6f} | "
              f"Gen: {avg_gen:.6f} | Val: {val_loss:.6f} | "
              f"Time: {elapsed:.1f}s | Total: {total_elapsed:.0f}s")

    total_time = time.time() - total_start
    print(f"\n  Training completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Final Recon Loss:   {history['recon_losses'][-1]:.6f}")
    print(f"  Final Val Loss:     {history['val_losses'][-1]:.6f}")
    print(f"  Final Disc Loss:    {history['disc_losses'][-1]:.6f}")
    print(f"  Final Gen Loss:     {history['gen_losses'][-1]:.6f}")

    return model, history


def _compute_val_loss(model, test_rgb, mse_loss, batch_size):
    total_loss = 0.0
    n = len(test_rgb)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        x_batch = test_rgb[i:end]
        x_hat, _ = model.forward(x_batch)
        loss = mse_loss.forward(x_hat, x_batch)
        total_loss += loss * (end - i)
    return total_loss / n
