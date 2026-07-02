import numpy as np


def evaluate_cgan(generator, latent_dim, num_classes=10, samples_per_class=10):
    print("\n" + "=" * 70)
    print("  cGAN EVALUATION")
    print("=" * 70)

    generator.set_training(False)
    all_samples = {}
    stats = {}

    for cls in range(num_classes):
        z = np.random.randn(samples_per_class, latent_dim)
        labels = np.full(samples_per_class, cls, dtype=int)
        generated = generator.forward(z, labels)
        generated = np.clip((generated + 1.0) / 2.0, 0.0, 1.0)
        generated = generated.reshape(-1, 28, 28)
        all_samples[cls] = generated

        mean_val = generated.mean()
        std_val = generated.std()
        stats[cls] = {
            'mean_pixel': mean_val,
            'std_pixel': std_val,
            'min_pixel': generated.min(),
            'max_pixel': generated.max(),
        }

    from datasets.fashion_mnist import CLASS_NAMES
    print(f"\n  {'Class':<15} | {'Mean':>7} | {'Std':>7} | {'Min':>7} | {'Max':>7}")
    print("  " + "-" * 55)
    for cls in range(num_classes):
        s = stats[cls]
        print(f"  {CLASS_NAMES[cls]:<15} | {s['mean_pixel']:7.4f} | "
              f"{s['std_pixel']:7.4f} | {s['min_pixel']:7.4f} | {s['max_pixel']:7.4f}")

    return all_samples, stats


def compute_diversity_score(samples):
    flat = samples.reshape(samples.shape[0], -1)
    dists = []
    n = min(len(flat), 20)
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(flat[i] - flat[j]))
    return np.mean(dists) if dists else 0.0
