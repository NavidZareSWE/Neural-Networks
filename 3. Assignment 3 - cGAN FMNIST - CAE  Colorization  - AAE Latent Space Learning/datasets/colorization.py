import numpy as np
import os
from PIL import Image


def load_colorization_dataset(data_dir, image_size=64):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    print(f"  Loading colorization dataset from: {data_dir}")
    print(f"  Target image size: {image_size}x{image_size}")

    train_rgb, _ = _load_images(train_dir, image_size)
    test_rgb, test_names = _load_images(test_dir, image_size)

    train_gray = rgb_to_grayscale(train_rgb)
    test_gray = rgb_to_grayscale(test_rgb)

    print(f"  Train set: {train_rgb.shape[0]} images")
    print(f"  Test set:  {test_rgb.shape[0]} images")
    print(f"  Train RGB shape:  {train_rgb.shape}")
    print(f"  Train Gray shape: {train_gray.shape}")
    print(f"  Pixel range: [{train_rgb.min():.3f}, {train_rgb.max():.3f}]")

    return train_gray, train_rgb, test_gray, test_rgb, test_names


def _load_images(directory, image_size):
    images = []
    names = []
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    filenames = sorted(os.listdir(directory))
    for fname in filenames:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_ext:
            continue

        path = os.path.join(directory, fname)
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((image_size, image_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float64) / 255.0
            arr = arr.transpose(2, 0, 1)
            images.append(arr)
            names.append(fname)
        except Exception as e:
            print(f"    Warning: Could not load {fname}: {e}")

    return np.array(images), names


def rgb_to_grayscale(rgb_images):
    r = rgb_images[:, 0:1, :, :]
    g = rgb_images[:, 1:2, :, :]
    b = rgb_images[:, 2:3, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
