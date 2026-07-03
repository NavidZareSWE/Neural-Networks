from utils import visualize
from utils.io_utils import ensure_dirs, save_metrics, save_training_history
from utils.helpers import set_seed, print_separator, print_model_summary
import sys
import os
import time
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def run_cgan(data_dir):
    from datasets.fashion_mnist import load_fashion_mnist, CLASS_NAMES
    from training.train_cgan import train_cgan
    from evaluation.eval_cgan import evaluate_cgan

    print_separator("PART 1: CONDITIONAL GAN ON FASHION-MNIST")

    # ---- Load Dataset ----
    print("\n[1/4] Loading Fashion-MNIST dataset...")
    images, labels = load_fashion_mnist(
        data_dir=os.path.join(data_dir, 'FashionMNIST'),
        split='train',
        normalize_range='tanh',
        max_samples=15000
    )

    visualize.plot_fashion_mnist_samples(images, labels, CLASS_NAMES)

    # ---- Print Architecture ----
    print("\n[2/4] Initializing cGAN architecture...")
    print("  Generator Architecture:")
    print("    Input: z(100) + one_hot(10) = 110")
    print("    Dense(110 -> 256) + BN + LeakyReLU(0.2)")
    print("    Dense(256 -> 512) + BN + LeakyReLU(0.2)")
    print("    Dense(512 -> 1024) + BN + LeakyReLU(0.2)")
    print("    Dense(1024 -> 784) + Tanh")
    print("  Discriminator Architecture:")
    print("    Input: image(784) + one_hot(10) = 794")
    print("    Dense(794 -> 512) + LeakyReLU(0.2) + Dropout(0.3)")
    print("    Dense(512 -> 256) + LeakyReLU(0.2) + Dropout(0.3)")
    print("    Dense(256 -> 1) + Sigmoid")

    # ---- Training ----
    print("\n[3/4] Training Conditional GAN...")
    config = {
        'latent_dim': 100,
        'num_classes': 10,
        'batch_size': 128,
        'num_epochs': 15,
        'lr_g': 0.0002,
        'lr_d': 0.0002,
        'beta1': 0.5,
        'save_interval': 3,
        'output_dir': 'output',
    }

    generator, discriminator, history = train_cgan(images, labels, config)

    print_model_summary(generator.get_layers(), "Generator")
    print_model_summary(discriminator.get_layers(), "Discriminator")

    # ---- Evaluation ----
    print("\n[4/4] Evaluating and generating final samples...")
    final_samples, stats = evaluate_cgan(
        generator, config['latent_dim'], config['num_classes'],
        samples_per_class=10
    )

    # ---- Save Plots ----
    print("\n  Generating plots...")
    visualize.plot_cgan_losses(history)
    visualize.plot_cgan_loss_components(history)

    for epoch, samples in history['samples'].items():
        visualize.plot_cgan_samples_grid(samples, CLASS_NAMES, epoch)

    final_grid = [final_samples[cls] for cls in range(10)]
    visualize.plot_cgan_samples_grid(final_grid, CLASS_NAMES, config['num_epochs'],
                                     filename='cgan_final_samples.png')

    visualize.plot_cgan_training_progression(history['samples'], CLASS_NAMES)

    save_training_history(history, 'output/training/cgan_history.npz')
    save_metrics(stats, 'output/evaluation/cgan_stats.json')

    print("\n  Part 1 (cGAN) completed successfully!")
    return generator, discriminator, history


def run_autoencoder(data_dir):
    from datasets.colorization import load_colorization_dataset
    from training.train_autoencoder import train_autoencoder
    from evaluation.eval_autoencoder import evaluate_autoencoder

    print_separator("PART 2: CONVOLUTIONAL AUTOENCODER FOR IMAGE COLORIZATION")

    # ---- Load Dataset ----
    print("\n[1/4] Loading colorization dataset...")
    image_size = 64
    ae_data_dir = os.path.join(data_dir, 'AE Data')
    train_gray, train_rgb, test_gray, test_rgb, test_names = load_colorization_dataset(
        data_dir=ae_data_dir,
        image_size=image_size
    )

    visualize.plot_colorization_samples(train_rgb)

    # ---- Print Architecture ----
    print("\n[2/4] Architecture details:")
    print("  Encoder:")
    print(
        f"    Conv2D(1->16, 3x3, pad=1) + BN + ReLU + MaxPool(2) -> {image_size//2}x{image_size//2}x16")
    print(
        f"    Conv2D(16->32, 3x3, pad=1) + BN + ReLU + MaxPool(2) -> {image_size//4}x{image_size//4}x32")
    print(
        f"    Conv2D(32->64, 3x3, pad=1) + BN + ReLU + MaxPool(2) -> {image_size//8}x{image_size//8}x64")
    print("  Decoder:")
    print(
        f"    Upsample(2) + Conv2D(64->32, 3x3, pad=1) + BN + ReLU -> {image_size//4}x{image_size//4}x32")
    print(
        f"    Upsample(2) + Conv2D(32->16, 3x3, pad=1) + BN + ReLU -> {image_size//2}x{image_size//2}x16")
    print(
        f"    Upsample(2) + Conv2D(16->3, 3x3, pad=1) + Sigmoid   -> {image_size}x{image_size}x3")

    # ---- Training ----
    print("\n[3/4] Training Convolutional Autoencoder...")
    config = {
        'batch_size': 8,
        'num_epochs': 10,
        'lr': 0.001,
        'output_dir': 'output',
    }

    model, history = train_autoencoder(
        train_gray, train_rgb, test_gray, test_rgb, config)

    # ---- Evaluation ----
    print("\n[4/4] Evaluating on test set...")
    predictions, metrics = evaluate_autoencoder(
        model, test_gray, test_rgb, test_names)

    # ---- Save Plots ----
    print("\n  Generating plots...")
    visualize.plot_ae_losses(history)
    visualize.plot_ae_test_results(
        test_gray, predictions, test_rgb, test_names)
    visualize.plot_ae_metrics_bar(metrics)

    save_training_history(history, 'output/training/ae_history.npz')
    save_metrics(metrics, 'output/evaluation/ae_metrics.json')

    print("\n  Part 2 (Autoencoder) completed successfully!")
    return model, history, metrics


def run_aae(data_dir):
    from datasets.colorization import load_colorization_dataset
    from training.train_aae import train_aae
    from evaluation.eval_aae import (evaluate_aae, encode_dataset,
                                     generate_from_prior, interpolate_latent)

    print_separator("BONUS: ADVERSARIAL AUTOENCODER FOR LATENT SPACE LEARNING")

    # ---- Load Dataset (RGB images) ----
    print("\n[1/6] Loading RGB dataset...")
    image_size = 64
    ae_data_dir = os.path.join(data_dir, 'AE Data')
    _, train_rgb, _, test_rgb, test_names = load_colorization_dataset(
        data_dir=ae_data_dir,
        image_size=image_size
    )

    # ---- Training ----
    print("\n[2/6] Training Adversarial Autoencoder...")
    config = {
        'batch_size': 8,
        'num_epochs': 8,
        'lr': 0.001,
        'latent_dim': 64,
        'output_dir': 'output',
    }

    model, history = train_aae(train_rgb, test_rgb, config)

    # ---- Evaluation: Reconstruction ----
    print("\n[3/6] Evaluating reconstruction on test set...")
    predictions, test_latents, metrics = evaluate_aae(
        model, test_rgb, test_names)

    # ---- Latent Space Analysis ----
    print("\n[4/6] Analyzing latent space distribution...")
    train_latents = encode_dataset(model, train_rgb, batch_size=8)
    print(f"  Encoded train set: {train_latents.shape}")
    print(f"  Latent mean (per-dim avg): {train_latents.mean():.4f}")
    print(f"  Latent std  (per-dim avg): {train_latents.std():.4f}")

    visualize.plot_aae_latent_pca(train_latents, test_latents)
    visualize.plot_aae_latent_histograms(train_latents)

    # ---- Generate from Prior ----
    print("\n[5/6] Generating samples from prior N(0, I)...")
    samples, _ = generate_from_prior(model, n_samples=10)
    visualize.plot_aae_samples(samples)

    # ---- Interpolation ----
    print("\n[6/6] Performing latent space interpolation...")
    z1 = test_latents[0]
    z2 = test_latents[1]
    interp_images = interpolate_latent(model, z1, z2, n_steps=8)
    visualize.plot_aae_interpolation(
        test_rgb[0], test_rgb[1], interp_images,
        name1=test_names[0], name2=test_names[1]
    )

    if len(test_rgb) >= 5:
        z3 = test_latents[0]
        z4 = test_latents[4]
        interp2 = interpolate_latent(model, z3, z4, n_steps=8)
        visualize.plot_aae_interpolation(
            test_rgb[0], test_rgb[4], interp2,
            name1=test_names[0], name2=test_names[4],
            filename='aae_interpolation_2.png'
        )

    # ---- Save Plots and Metrics ----
    visualize.plot_aae_losses(history)
    visualize.plot_aae_reconstruction(test_rgb, predictions, test_names)

    save_training_history(history, 'output/training/aae_history.npz')
    save_metrics(metrics, 'output/evaluation/aae_metrics.json')

    print("\n  Bonus (Adversarial Autoencoder) completed successfully!")
    return model, history, metrics


def main():
    start_time = time.time()

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   Neural Networks and Deep Learning - Assignment 3              #")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    set_seed(42)

    ensure_dirs('output')

    possible_dirs = [
        'data',
        '../project_data/data',
        os.path.expanduser('~/project_data/data'),
    ]

    data_dir = None
    for d in possible_dirs:
        if os.path.exists(d):
            data_dir = d
            break

    if data_dir is None:
        print("\n  ERROR: Data directory not found!")
        print(
            "  Expected 'data/' directory with FashionMNIST/ and AE Data/ subdirectories.")
        print("  Please extract data.zip to a 'data/' directory in the project root.")
        sys.exit(1)

    print(f"\n  Data directory: {os.path.abspath(data_dir)}")

    task = 'all'
    if len(sys.argv) > 1:
        if sys.argv[1] == '--task' and len(sys.argv) > 2:
            task = sys.argv[2].lower()
        else:
            task = sys.argv[1].lower()

    if task in ('all', 'cgan', '1'):
        run_cgan(data_dir)

    if task in ('all', 'autoencoder', 'ae', '2'):
        run_autoencoder(data_dir)

    if task in ('all', 'aae', 'bonus', '3'):
        run_aae(data_dir)

    total_time = time.time() - start_time
    print("\n" + "#" * 70)
    print(
        f"#  All tasks completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print("#" * 70)
    print("\n  Output files:")
    for root, dirs, files in os.walk('output'):
        for f in sorted(files):
            filepath = os.path.join(root, f)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"    {filepath} ({size_kb:.1f} KB)")
    print()


if __name__ == '__main__':
    main()
