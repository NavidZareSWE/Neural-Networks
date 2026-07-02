import numpy as np


def compute_ssim_gray(img1, img2):
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim)


def evaluate_aae(model, test_rgb, test_names):
    print("\n" + "=" * 70)
    print("  ADVERSARIAL AUTOENCODER EVALUATION")
    print("=" * 70)

    model.set_training(False)
    predictions, latents = [], []

    for i in range(len(test_rgb)):
        x = test_rgb[i:i+1]
        x_hat, z = model.forward(x)
        predictions.append(x_hat[0])
        latents.append(z[0])

    predictions = np.array(predictions)
    latents = np.array(latents)

    per_image = []
    print(f"\n  {'Image':<22s} | {'MSE':>10s} | {'PSNR (dB)':>10s} | {'SSIM':>10s}")
    print("  " + "-" * 60)

    for i in range(len(test_rgb)):
        pred = predictions[i]
        target = test_rgb[i]
        mse = float(np.mean((pred - target) ** 2))
        psnr = float(10 * np.log10(1.0 / (mse + 1e-10)))

        ssim_vals = []
        for c in range(3):
            ssim_vals.append(compute_ssim_gray(pred[c], target[c]))
        ssim = float(np.mean(ssim_vals))

        name = test_names[i] if test_names else f"image_{i}"
        per_image.append({'name': name, 'mse': mse, 'psnr': psnr, 'ssim': ssim})
        print(f"  {name:<22s} | {mse:10.6f} | {psnr:10.2f} | {ssim:10.4f}")

    avg_mse = np.mean([m['mse'] for m in per_image])
    avg_psnr = np.mean([m['psnr'] for m in per_image])
    avg_ssim = np.mean([m['ssim'] for m in per_image])

    print("  " + "-" * 60)
    print(f"  {'AVERAGE':<22s} | {avg_mse:10.6f} | {avg_psnr:10.2f} | {avg_ssim:10.4f}")

    metrics = {
        'per_image': per_image,
        'avg_mse': float(avg_mse),
        'avg_psnr': float(avg_psnr),
        'avg_ssim': float(avg_ssim),
    }

    return predictions, latents, metrics


def encode_dataset(model, data, batch_size=8):
    model.set_training(False)
    all_z = []
    for i in range(0, len(data), batch_size):
        end = min(i + batch_size, len(data))
        z = model.encode(data[i:end])
        all_z.append(z)
    return np.concatenate(all_z, axis=0)


def generate_from_prior(model, n_samples=10):
    model.set_training(False)
    z = np.random.randn(n_samples, model.latent_dim).astype(np.float32)
    images = model.decode(z)
    return images, z


def interpolate_latent(model, z1, z2, n_steps=8):
    model.set_training(False)
    alphas = np.linspace(0, 1, n_steps)
    z_interp = np.array([z1 * (1 - a) + z2 * a for a in alphas])
    images = model.decode(z_interp.astype(np.float32))
    return images
