# Neural Networks and Deep Learning - Assignment 3

## Project Structure

```

├── main.py                         # Entry point
├── requirements.txt                # Dependencies
├── README.md                       # This file
│
├── models/
│   ├── layers.py                   # Dense, Conv2D, MaxPool, Upsample, BatchNorm, etc.
│   ├── activations.py              # ReLU, LeakyReLU, Sigmoid, Tanh
│   ├── optimizers.py               # Adam, SGD
│   ├── losses.py                   # BCELoss, MSELoss
│   ├── cgan.py                     # Generator and Discriminator for cGAN
│   └── conv_autoencoder.py         # Convolutional Autoencoder
│
├── training/
│   ├── train_cgan.py               # cGAN training loop
│   └── train_autoencoder.py        # Autoencoder training loop
│
├── evaluation/
│   ├── eval_cgan.py                # cGAN evaluation and sample generation
│   └── eval_autoencoder.py         # Autoencoder metrics (MSE, PSNR, SSIM)
│
├── datasets/
│   ├── fashion_mnist.py            # Fashion-MNIST loader (raw IDX format)
│   └── colorization.py             # Colorization dataset loader
│
├── utils/
│   ├── visualize.py                # All plotting functions
│   ├── helpers.py                  # Seed, timing, model summary
│   └── io_utils.py                 # Save/load metrics and history
│
├── output/
│   ├── training/                   # Training histories
│   ├── evaluation/                 # Metrics JSON files
│   └── plots/                      # All generated figures
│
└── data/                           # Dataset (extract data.zip here)
    ├── FashionMNIST/raw/           # Fashion-MNIST IDX files
    └── AE Data/
        ├── train/                  # 490 training images
        └── test/                   # 10 test images
```

## Setup

1. Extract `data.zip` into a `data/` directory at the project root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running

```bash
# Run both tasks
python main.py

# Run only Part 1 (cGAN)
python main.py --task cgan

# Run only Part 2 (Autoencoder)
python main.py --task autoencoder
```

## Implementation Details

### Part 1: Conditional GAN (cGAN)
- **Architecture**: MLP-based Generator and Discriminator
- **Conditioning**: One-hot class vectors (num_classes dims) concatenated with the noise/image inputs
- **Loss**: Binary Cross-Entropy with hard real/fake targets (1.0/0.0)
- **Optimizer**: Adam (lr=0.0002, beta1=0.5)
- **All layers implemented from scratch** in NumPy

### Part 2: Convolutional Autoencoder
- **Architecture**: Encoder-Decoder with Conv2D, MaxPool, Upsample
- **Input**: 64x64 grayscale images (1 channel)
- **Output**: 64x64 RGB images (3 channels)
- **Loss**: Mean Squared Error (MSE)
- **Metrics**: MSE, PSNR, SSIM
- **All convolutions implemented from scratch** using im2col/col2im

## Dependencies
- NumPy (matrix operations)
- Matplotlib (visualization)
- Pillow (image loading)
- scikit-image (SSIM metric)
- SciPy (scientific computing)
