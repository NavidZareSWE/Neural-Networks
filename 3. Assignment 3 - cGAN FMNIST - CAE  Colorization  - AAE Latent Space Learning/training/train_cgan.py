import numpy as np
import time
import os
from models.cgan import Generator, Discriminator
from models.optimizers import Adam
from models.losses import BCELoss


def train_cgan(images, labels, config):
    latent_dim = config.get('latent_dim', 100)
    num_classes = config.get('num_classes', 10)
    batch_size = config.get('batch_size', 128)
    num_epochs = config.get('num_epochs', 15)
    lr_g = config.get('lr_g', 0.0002)
    lr_d = config.get('lr_d', 0.0002)
    beta1 = config.get('beta1', 0.5)
    save_interval = config.get('save_interval', 3)

    print("=" * 70)
    print("  CONDITIONAL GAN TRAINING")
    print("=" * 70)
    print(f"  Dataset size:    {len(images)} samples")
    print(f"  Latent dim:      {latent_dim}")
    print(f"  Conditioning:    one-hot ({num_classes} classes)")
    print(f"  Batch size:      {batch_size}")
    print(f"  Epochs:          {num_epochs}")
    print(f"  Learning rate G: {lr_g}")
    print(f"  Learning rate D: {lr_d}")
    print(f"  Adam beta1:      {beta1}")
    print("=" * 70)

    generator = Generator(latent_dim=latent_dim, num_classes=num_classes)
    discriminator = Discriminator(num_classes=num_classes)

    opt_g = Adam(lr=lr_g, beta1=beta1)
    opt_d = Adam(lr=lr_d, beta1=beta1)

    bce = BCELoss()

    history = {
        'g_losses': [],
        'd_losses': [],
        'd_real_losses': [],
        'd_fake_losses': [],
        'epoch_g_losses': [],
        'epoch_d_losses': [],
        'samples': {},
    }

    n_samples = len(images)
    n_batches = n_samples // batch_size
    total_start = time.time()

    real_label_val = 1.0
    fake_label_val = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        perm = np.random.permutation(n_samples)
        images_shuffled = images[perm]
        labels_shuffled = labels[perm]

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        generator.set_training(True)
        discriminator.set_training(True)

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            real_imgs = images_shuffled[start:end]
            real_lbls = labels_shuffled[start:end]
            bs = real_imgs.shape[0]

            # ----------------------------------------
            # Train Discriminator
            # ----------------------------------------
            # Real samples
            d_real_out = discriminator.forward(real_imgs, real_lbls)
            d_real_loss = bce.forward(
                d_real_out, np.full((bs, 1), real_label_val))
            d_real_grad = bce.backward()
            discriminator.backward(d_real_grad)

            d_real_grads_saved = []
            for layer in discriminator.get_layers():
                for p, g, n in layer.params_and_grads():
                    d_real_grads_saved.append(g.copy())

            z = np.random.randn(bs, latent_dim)
            fake_lbls = np.random.randint(0, num_classes, bs)
            fake_imgs = generator.forward(z, fake_lbls)

            d_fake_out = discriminator.forward(fake_imgs, fake_lbls)
            d_fake_loss = bce.forward(
                d_fake_out, np.full((bs, 1), fake_label_val))
            d_fake_grad = bce.backward()
            discriminator.backward(d_fake_grad)

            idx = 0
            for layer in discriminator.get_layers():
                for p, g, n in layer.params_and_grads():
                    g += d_real_grads_saved[idx]
                    idx += 1

            d_loss = d_real_loss + d_fake_loss
            opt_d.step(discriminator.get_layers())

            # ----------------------------------------
            # Train Generator
            # ----------------------------------------
            z = np.random.randn(bs, latent_dim)
            gen_lbls = np.random.randint(0, num_classes, bs)

            fake_imgs = generator.forward(z, gen_lbls)

            d_out = discriminator.forward(fake_imgs, gen_lbls)
            g_loss = bce.forward(d_out, np.ones((bs, 1)))
            g_grad = bce.backward()

            d_img_grad = discriminator.backward(g_grad)

            generator.backward(d_img_grad)
            opt_g.step(generator.get_layers())

            history['g_losses'].append(g_loss)
            history['d_losses'].append(d_loss)
            history['d_real_losses'].append(d_real_loss)
            history['d_fake_losses'].append(d_fake_loss)

            epoch_g_loss += g_loss
            epoch_d_loss += d_loss

        avg_g = epoch_g_loss / n_batches
        avg_d = epoch_d_loss / n_batches
        history['epoch_g_losses'].append(avg_g)
        history['epoch_d_losses'].append(avg_d)

        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - total_start

        print(f"  Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"D_loss: {avg_d:.4f} | G_loss: {avg_g:.4f} | "
              f"D_real: {np.mean(history['d_real_losses'][-n_batches:]):.4f} | "
              f"D_fake: {np.mean(history['d_fake_losses'][-n_batches:]):.4f} | "
              f"Time: {elapsed:.1f}s | Total: {total_elapsed:.0f}s")

        if (epoch + 1) % save_interval == 0 or epoch == 0 or (epoch + 1) == num_epochs:
            generator.set_training(False)
            samples = generate_grid_samples(generator, latent_dim, num_classes)
            history['samples'][epoch + 1] = samples
            generator.set_training(True)
            print(f"    -> Saved sample grid at epoch {epoch+1}")

    total_time = time.time() - total_start
    print(
        f"\n  Training completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Final G_loss: {history['epoch_g_losses'][-1]:.4f}")
    print(f"  Final D_loss: {history['epoch_d_losses'][-1]:.4f}")

    return generator, discriminator, history


def generate_grid_samples(generator, latent_dim, num_classes, samples_per_class=8):
    all_samples = []
    for cls in range(num_classes):
        z = np.random.randn(samples_per_class, latent_dim)
        labels = np.full(samples_per_class, cls, dtype=int)
        generated = generator.forward(z, labels)
        generated = (generated + 1.0) / 2.0
        generated = generated.reshape(-1, 28, 28)
        all_samples.append(generated)
    return all_samples
