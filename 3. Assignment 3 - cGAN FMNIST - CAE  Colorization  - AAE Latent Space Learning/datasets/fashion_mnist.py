import numpy as np
import struct
import gzip
import os


CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def read_idx_images(filepath):
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rb') as f:
            data = f.read()
    else:
        with open(filepath, 'rb') as f:
            data = f.read()

    magic = struct.unpack('>I', data[:4])[0]
    n_images = struct.unpack('>I', data[4:8])[0]
    n_rows = struct.unpack('>I', data[8:12])[0]
    n_cols = struct.unpack('>I', data[12:16])[0]

    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    images = images.reshape(n_images, n_rows, n_cols)
    return images


def read_idx_labels(filepath):
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rb') as f:
            data = f.read()
    else:
        with open(filepath, 'rb') as f:
            data = f.read()

    magic = struct.unpack('>I', data[:4])[0]
    n_labels = struct.unpack('>I', data[4:8])[0]

    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return labels


def load_fashion_mnist(data_dir, split='train', normalize_range='tanh', max_samples=None):
    raw_dir = os.path.join(data_dir, 'raw')

    if split == 'train':
        img_file_base = 'train-images-idx3-ubyte'
        lbl_file_base = 'train-labels-idx1-ubyte'
    else:
        img_file_base = 't10k-images-idx3-ubyte'
        lbl_file_base = 't10k-labels-idx1-ubyte'

    img_path = os.path.join(raw_dir, img_file_base)
    lbl_path = os.path.join(raw_dir, lbl_file_base)
    if not os.path.exists(img_path):
        img_path += '.gz'
        lbl_path += '.gz'

    print(f"  Loading {split} images from: {img_path}")
    print(f"  Loading {split} labels from: {lbl_path}")

    images = read_idx_images(img_path)
    labels = read_idx_labels(lbl_path)

    if max_samples is not None and max_samples < len(images):
        indices = np.random.choice(len(images), max_samples, replace=False)
        images = images[indices]
        labels = labels[indices]

    images = images.reshape(-1, 784).astype(np.float64)
    if normalize_range == 'tanh':
        images = (images / 255.0) * 2.0 - 1.0
    else:
        images = images / 255.0

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    distribution = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, label_counts)}
    print(f"  Loaded {len(images)} {split} samples, shape: {images.shape}")
    print(f"  Label distribution: {distribution}")
    return images, labels
