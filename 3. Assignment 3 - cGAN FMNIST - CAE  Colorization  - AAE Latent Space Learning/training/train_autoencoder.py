import numpy as np
import time
from models.conv_autoencoder import ConvAutoencoder
from models.optimizers import Adam
from models.losses import MSELoss


def train_autoencoder(train_gray, train_rgb, test_gray, test_rgb, config):
    batch_size = config.get('batch_size', 8)
    num_epochs = config.get('num_epochs', 50)
    lr = config.get('lr', 0.001)
    output_dir = config.get('output_dir', 'output')

    print("=" * 70)
    print("  CONVOLUTIONAL AUTOENCODER TRAINING (Image Colorization)")
    print("=" * 70)
    print(f"  Training samples: {len(train_gray)}")
    print(f"  Test samples:     {len(test_gray)}")
    print(f"  Image size:       {train_gray.shape[2]}x{train_gray.shape[3]}")
    print(f"  Input channels:   {train_gray.shape[1]} (grayscale)")
    print(f"  Output channels:  {train_rgb.shape[1]} (RGB)")
    print(f"  Batch size:       {batch_size}")
    print(f"  Epochs:           {num_epochs}")
    print(f"  Learning rate:    {lr}")

    print("\n  Architecture:")
    print("    Encoder: Conv(1->16)+BN+ReLU+Pool -> Conv(16->32)+BN+ReLU+Pool -> Conv(32->64)+BN+ReLU+Pool")
    print("    Decoder: Up+Conv(64->32)+BN+ReLU -> Up+Conv(32->16)+BN+ReLU -> Up+Conv(16->3)+Sigmoid")
    print("  Loss: Mean Squared Error (MSE)")
    print("  Optimizer: Adam")
    print("=" * 70)

    model = ConvAutoencoder()
    optimizer = Adam(lr=lr, beta1=0.9, beta2=0.999)
    mse_loss = MSELoss()

    total_params = 0
    for layer in model.get_layers():
        for p, _, _ in layer.params_and_grads():
            total_params += p.size
    print(f"  Total trainable parameters: {total_params:,}")

    history = {
        'train_losses': [],
        'val_losses': [],
    }

    n_train = len(train_gray)
    n_batches = max(1, n_train // batch_size)
    total_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.set_training(True)

        perm = np.random.permutation(n_train)
        train_gray_s = train_gray[perm]
        train_rgb_s = train_rgb[perm]

        epoch_loss = 0.0
        n_processed = 0

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_train)
            gray_batch = train_gray_s[start:end]
            rgb_batch = train_rgb_s[start:end]

            pred_rgb = model.forward(gray_batch)

            loss = mse_loss.forward(pred_rgb, rgb_batch)

            grad = mse_loss.backward()
            model.backward(grad)

            optimizer.step(model.get_layers())

            epoch_loss += loss * (end - start)
            n_processed += (end - start)

        avg_train_loss = epoch_loss / n_processed

        model.set_training(False)
        val_loss = _compute_val_loss(model, test_gray, test_rgb, mse_loss, batch_size)

        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(val_loss)

        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - total_start

        if (epoch + 1) % 2 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
            print(f"  Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Time: {elapsed:.1f}s | Total: {total_elapsed:.0f}s")

    total_time = time.time() - total_start
    print(f"\n  Training completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Final Train Loss: {history['train_losses'][-1]:.6f}")
    print(f"  Final Val Loss:   {history['val_losses'][-1]:.6f}")
    best_epoch = np.argmin(history['val_losses']) + 1
    print(f"  Best Val Loss:    {min(history['val_losses']):.6f} at epoch {best_epoch}")

    return model, history


def _compute_val_loss(model, test_gray, test_rgb, mse_loss, batch_size):
    total_loss = 0.0
    n = len(test_gray)

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        gray_batch = test_gray[i:end]
        rgb_batch = test_rgb[i:end]
        pred_rgb = model.forward(gray_batch)
        loss = mse_loss.forward(pred_rgb, rgb_batch)
        total_loss += loss * (end - i)

    return total_loss / n
