import numpy as np
from skimage.metrics import structural_similarity as ssim


def evaluate_autoencoder(model, test_gray, test_rgb, test_names):
    print("\n" + "=" * 70)
    print("  AUTOENCODER EVALUATION")
    print("=" * 70)

    model.set_training(False)

    predictions = model.forward(test_gray)
    predictions = np.clip(predictions, 0.0, 1.0)

    per_image_metrics = []
    print(f"\n  {'Image':<20} | {'MSE':>10} | {'PSNR (dB)':>10} | {'SSIM':>10}")
    print("  " + "-" * 60)

    for i in range(len(test_gray)):
        pred = predictions[i]
        target = test_rgb[i]

        pred_hwc = pred.transpose(1, 2, 0)
        target_hwc = target.transpose(1, 2, 0)

        mse = np.mean((pred_hwc - target_hwc) ** 2)

        if mse > 0:
            psnr = 10.0 * np.log10(1.0 / mse)
        else:
            psnr = float('inf')

        ssim_val = ssim(target_hwc, pred_hwc,
                        data_range=1.0,
                        channel_axis=2)

        per_image_metrics.append({
            'name': test_names[i],
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim_val,
        })

        print(f"  {test_names[i]:<20} | {mse:10.6f} | {psnr:10.2f} | {ssim_val:10.4f}")

    avg_mse = np.mean([m['mse'] for m in per_image_metrics])
    avg_psnr = np.mean([m['psnr'] for m in per_image_metrics])
    avg_ssim = np.mean([m['ssim'] for m in per_image_metrics])

    print("  " + "-" * 60)
    print(f"  {'AVERAGE':<20} | {avg_mse:10.6f} | {avg_psnr:10.2f} | {avg_ssim:10.4f}")

    metrics = {
        'per_image': per_image_metrics,
        'avg_mse': avg_mse,
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
    }

    best_idx = np.argmax([m['ssim'] for m in per_image_metrics])
    worst_idx = np.argmin([m['ssim'] for m in per_image_metrics])
    print(f"\n  Best colorized:  {per_image_metrics[best_idx]['name']} (SSIM={per_image_metrics[best_idx]['ssim']:.4f})")
    print(f"  Worst colorized: {per_image_metrics[worst_idx]['name']} (SSIM={per_image_metrics[worst_idx]['ssim']:.4f})")

    return predictions, metrics
