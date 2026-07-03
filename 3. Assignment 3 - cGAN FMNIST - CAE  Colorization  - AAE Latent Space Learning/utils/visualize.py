import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')


PLOT_DIR = 'output/plots'


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _savefig(fig, filename, dpi=150):
    _ensure_dir(PLOT_DIR)
    filepath = os.path.join(PLOT_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"    [SAVED] {filepath}")


def _suptitle(fig, title, fontsize=14):
    h = fig.get_size_inches()[1]
    reserve = 0.7
    fig.tight_layout(rect=[0, 0, 1, 1 - reserve / h])
    fig.suptitle(title, fontsize=fontsize, fontweight='bold', y=1 - 0.30 / h)


# ------------------------------------------------------------
# cGAN Visualizations
# ------------------------------------------------------------

def plot_cgan_losses(history, filename='cgan_losses.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(history['g_losses'], alpha=0.3,
            color='blue', label='G Loss (iter)')
    ax.plot(history['d_losses'], alpha=0.3, color='red', label='D Loss (iter)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Per-Iteration Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    epochs = range(1, len(history['epoch_g_losses']) + 1)
    ax.plot(epochs, history['epoch_g_losses'], 'b-o',
            markersize=3, label='G Loss (epoch avg)')
    ax.plot(epochs, history['epoch_d_losses'], 'r-o',
            markersize=3, label='D Loss (epoch avg)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.set_title('Per-Epoch Average Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    _suptitle(fig, 'Conditional GAN Training Losses', fontsize=14)
    _savefig(fig, filename)


def plot_cgan_loss_components(history, filename='cgan_loss_components.png'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    window = max(1, len(history['d_real_losses']) // 100)
    d_real_smooth = _moving_average(history['d_real_losses'], window)
    d_fake_smooth = _moving_average(history['d_fake_losses'], window)

    ax.plot(d_real_smooth, label='D Loss (Real)', color='green', alpha=0.8)
    ax.plot(d_fake_smooth, label='D Loss (Fake)', color='orange', alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Discriminator Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig(fig, filename)


def plot_cgan_samples_grid(samples, class_names, epoch, filename=None):
    if filename is None:
        filename = f'cgan_samples_epoch_{epoch:03d}.png'

    num_classes = len(samples)
    samples_per_class = samples[0].shape[0]

    fig, axes = plt.subplots(num_classes, samples_per_class,
                             figsize=(samples_per_class * 1.2, num_classes * 1.4))

    for row in range(num_classes):
        for col in range(samples_per_class):
            ax = axes[row, col]
            img = np.clip(samples[row][col], 0, 1)
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(class_names[row], fontsize=7, rotation=0,
                              labelpad=50, va='center')

    _suptitle(fig, f'Generated Samples - Epoch {epoch}', fontsize=12)
    _savefig(fig, filename)


def plot_cgan_training_progression(history_samples, class_names, filename='cgan_progression.png'):
    epochs = sorted(history_samples.keys())
    if len(epochs) < 2:
        return

    n_epochs_show = min(len(epochs), 5)
    epoch_indices = np.linspace(0, len(epochs) - 1, n_epochs_show, dtype=int)
    selected_epochs = [epochs[i] for i in epoch_indices]
    classes_to_show = [0, 1, 5, 7, 8]

    fig, axes = plt.subplots(len(classes_to_show), n_epochs_show,
                             figsize=(n_epochs_show * 2, len(classes_to_show) * 2))

    for row, cls in enumerate(classes_to_show):
        for col, ep in enumerate(selected_epochs):
            ax = axes[row, col]
            samples = history_samples[ep]
            img = np.clip(samples[cls][0], 0, 1)
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if row == 0:
                ax.set_title(f'Epoch {ep}', fontsize=9)
            if col == 0:
                ax.set_ylabel(class_names[cls], fontsize=8, rotation=0,
                              labelpad=45, va='center')

    _suptitle(
        fig, 'Training Progression: Generated Samples Over Epochs', fontsize=12)
    _savefig(fig, filename)


# ------------------------------------------------------------
# Autoencoder Visualizations
# ------------------------------------------------------------

def plot_ae_losses(history, filename='ae_losses.png'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    epochs = range(1, len(history['train_losses']) + 1)
    ax.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Convolutional Autoencoder: Training vs Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig(fig, filename)


def plot_ae_test_results(test_gray, predictions, test_rgb, test_names,
                         filename='ae_test_results.png'):
    n = len(test_gray)
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 2.5))

    col_titles = ['Grayscale Input', 'Colorized Output', 'Original RGB']

    for i in range(n):
        gray_img = test_gray[i, 0]
        axes[i, 0].imshow(gray_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_ylabel(test_names[i], fontsize=7, rotation=0,
                              labelpad=55, va='center')

        pred_img = np.clip(predictions[i].transpose(1, 2, 0), 0, 1)
        axes[i, 1].imshow(pred_img)

        orig_img = test_rgb[i].transpose(1, 2, 0)
        axes[i, 2].imshow(orig_img)

        for j in range(3):
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(
                    col_titles[j], fontsize=10, fontweight='bold')

    _suptitle(fig, 'Image Colorization Results on Test Set', fontsize=13)
    _savefig(fig, filename)


def plot_ae_metrics_bar(metrics, filename='ae_metrics_bar.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = [m['name'][:12] for m in metrics['per_image']]
    psnr_vals = [m['psnr'] for m in metrics['per_image']]
    ssim_vals = [m['ssim'] for m in metrics['per_image']]

    ax = axes[0]
    bars = ax.bar(range(len(names)), psnr_vals, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Peak Signal-to-Noise Ratio per Test Image')
    ax.axhline(y=np.mean(psnr_vals), color='red', linestyle='--', alpha=0.7,
               label=f'Average: {np.mean(psnr_vals):.2f} dB')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    bars = ax.bar(range(len(names)), ssim_vals, color='coral', alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('SSIM')
    ax.set_title('Structural Similarity Index per Test Image')
    ax.axhline(y=np.mean(ssim_vals), color='red', linestyle='--', alpha=0.7,
               label=f'Average: {np.mean(ssim_vals):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    _savefig(fig, filename)


# ------------------------------------------------------------
# Dataset Visualization
# ------------------------------------------------------------

def plot_fashion_mnist_samples(images, labels, class_names,
                               filename='fashion_mnist_samples.png'):
    fig, axes = plt.subplots(2, 10, figsize=(15, 3.5))
    for cls in range(10):
        idx = np.where(labels == cls)[0]
        for row in range(2):
            ax = axes[row, cls]
            img = images[idx[row]].reshape(28, 28)
            img = (img + 1.0) / 2.0
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if row == 0:
                ax.set_title(class_names[cls], fontsize=7)

    _suptitle(fig, 'Fashion-MNIST Dataset Samples', fontsize=12)
    _savefig(fig, filename)


def plot_colorization_samples(train_rgb, filename='colorization_samples.png'):
    n = min(20, len(train_rgb))
    cols = 5
    rows = n // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.2))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i, j]
            img = train_rgb[idx].transpose(1, 2, 0)
            ax.imshow(img)
            ax.axis('off')

    _suptitle(fig, 'Colorization Training Dataset Samples', fontsize=12)
    _savefig(fig, filename)


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def _moving_average(data, window):
    if window <= 1:
        return data
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    return cumsum[window - 1:] / window


def set_plot_dir(path):
    global PLOT_DIR
    PLOT_DIR = path


# -------------------------------------------------------------------------
#  Adversarial Autoencoder (AAE) Visualizations
# -------------------------------------------------------------------------

def plot_aae_losses(history, filename='aae_losses.png'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history['recon_losses']) + 1)

    axes[0].plot(epochs, history['recon_losses'], 'b-',
                 linewidth=2, label='Train Recon')
    axes[0].plot(epochs, history['val_losses'], 'r-',
                 linewidth=2, label='Val Recon')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Reconstruction Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['disc_losses'], 'g-',
                 linewidth=2, label='Discriminator')
    axes[1].axhline(y=np.log(2), color='gray', linestyle='--',
                    alpha=0.5, label='Optimal (ln2)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('BCE Loss')
    axes[1].set_title('Discriminator Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['gen_losses'], 'm-',
                 linewidth=2, label='Generator (Enc)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('BCE Loss')
    axes[2].set_title('Generator Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    _suptitle(fig, 'Adversarial Autoencoder Training Losses', fontsize=14)
    _savefig(fig, filename)


def plot_aae_reconstruction(test_rgb, predictions, test_names, filename='aae_reconstruction.png'):
    n = len(test_rgb)
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.5))

    for i in range(n):
        img_orig = np.transpose(test_rgb[i], (1, 2, 0))
        axes[0, i].imshow(np.clip(img_orig, 0, 1))
        axes[0, i].set_title(
            test_names[i] if test_names else f'img_{i}', fontsize=7)
        axes[0, i].axis('off')

        img_recon = np.transpose(predictions[i], (1, 2, 0))
        axes[1, i].imshow(np.clip(img_recon, 0, 1))
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=11)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=11)
    _suptitle(fig, 'AAE Reconstruction on Test Set', fontsize=14)
    _savefig(fig, filename)


def plot_aae_latent_pca(train_latents, test_latents=None, filename='aae_latent_pca.png'):
    mean = train_latents.mean(axis=0)
    centered = train_latents - mean
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    pc = centered @ eigenvectors[:, :2]

    n_prior = len(train_latents)
    z_prior = np.random.randn(n_prior, train_latents.shape[1])
    prior_centered = z_prior - mean
    pc_prior = prior_centered @ eigenvectors[:, :2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].scatter(pc[:, 0], pc[:, 1], alpha=0.3,
                    s=10, c='blue', label='Encoded')
    if test_latents is not None:
        tc = (test_latents - mean) @ eigenvectors[:, :2]
        axes[0].scatter(tc[:, 0], tc[:, 1], alpha=0.9, s=60, c='red',
                        marker='*', label='Test', zorder=5)
    axes[0].set_xlabel('PC 1')
    axes[0].set_ylabel('PC 2')
    axes[0].set_title('Encoded Latent Space (PCA)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(pc_prior[:, 0], pc_prior[:, 1], alpha=0.3,
                    s=10, c='green', label='Prior N(0,I)')
    axes[1].set_xlabel('PC 1')
    axes[1].set_ylabel('PC 2')
    axes[1].set_title('Prior Distribution (PCA)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    xlim = max(abs(pc[:, 0]).max(), abs(pc_prior[:, 0]).max()) * 1.2
    ylim = max(abs(pc[:, 1]).max(), abs(pc_prior[:, 1]).max()) * 1.2
    for ax in axes[:2]:
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)

    axes[2].scatter(pc_prior[:, 0], pc_prior[:, 1],
                    alpha=0.2, s=10, c='green', label='Prior')
    axes[2].scatter(pc[:, 0], pc[:, 1], alpha=0.2,
                    s=10, c='blue', label='Encoded')
    axes[2].set_xlabel('PC 1')
    axes[2].set_ylabel('PC 2')
    axes[2].set_title('Encoded vs Prior (Overlay)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(-xlim, xlim)
    axes[2].set_ylim(-ylim, ylim)

    _suptitle(fig, 'Latent Space Analysis via PCA', fontsize=14)
    _savefig(fig, filename)


def plot_aae_latent_histograms(train_latents, filename='aae_latent_histograms.png'):
    n_dims_to_show = min(8, train_latents.shape[1])
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    x_gauss = np.linspace(-4, 4, 200)
    y_gauss = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_gauss ** 2)

    for i in range(n_dims_to_show):
        ax = axes[i]
        values = train_latents[:, i]
        ax.hist(values, bins=30, density=True, alpha=0.7, color='steelblue',
                edgecolor='white', label='Encoded')
        ax.plot(x_gauss, y_gauss, 'r-', linewidth=2, label='N(0,1)')
        ax.set_title(
            f'z[{i}]  μ={values.mean():.2f}  σ={values.std():.2f}', fontsize=9)
        ax.set_xlim(-4, 4)
        if i == 0:
            ax.legend(fontsize=8)

    _suptitle(fig, 'Latent Dimension Histograms vs Gaussian Prior', fontsize=14)
    _savefig(fig, filename)


def plot_aae_samples(samples, filename='aae_prior_samples.png'):
    n = len(samples)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = axes.flatten()

    for i in range(n):
        img = np.transpose(samples[i], (1, 2, 0))
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].set_title(f'Sample {i+1}', fontsize=9)
        axes[i].axis('off')
    for i in range(n, len(axes)):
        axes[i].axis('off')

    _suptitle(fig, 'Generated Samples from Prior N(0, I)', fontsize=14)
    _savefig(fig, filename)


def plot_aae_interpolation(img1, img2, interp_images, name1='', name2='',
                           filename='aae_interpolation.png'):
    n_steps = len(interp_images)
    fig, axes = plt.subplots(1, n_steps + 2, figsize=(2.5 * (n_steps + 2), 3))

    axes[0].imshow(np.clip(np.transpose(img1, (1, 2, 0)), 0, 1))
    axes[0].set_title(f'Start\n{name1}', fontsize=8)
    axes[0].axis('off')

    for i in range(n_steps):
        img = np.transpose(interp_images[i], (1, 2, 0))
        axes[i + 1].imshow(np.clip(img, 0, 1))
        alpha = i / (n_steps - 1) if n_steps > 1 else 0
        axes[i + 1].set_title(f'α={alpha:.2f}', fontsize=8)
        axes[i + 1].axis('off')

    axes[-1].imshow(np.clip(np.transpose(img2, (1, 2, 0)), 0, 1))
    axes[-1].set_title(f'End\n{name2}', fontsize=8)
    axes[-1].axis('off')

    _suptitle(fig, 'Latent Space Interpolation', fontsize=14)
    _savefig(fig, filename)
