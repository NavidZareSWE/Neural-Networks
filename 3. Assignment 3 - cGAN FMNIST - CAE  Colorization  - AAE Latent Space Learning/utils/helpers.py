import numpy as np
import time
import os


def set_seed(seed=42):
    np.random.seed(seed)
    print(f"  Random seed set to: {seed}")


def print_separator(title=""):
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    else:
        print("=" * 70)


def print_model_summary(layers, name="Model"):
    total = 0
    print(f"\n  {name} Summary:")
    print(f"  {'Layer':<30} | {'Param Shape':<20} | {'Params':>10}")
    print("  " + "-" * 65)
    for layer in layers:
        for p, _, n in layer.params_and_grads():
            count = p.size
            total += count
            layer_name = f"{type(layer).__name__}.{n}"
            print(f"  {layer_name:<30} | {str(p.shape):<20} | {count:>10,}")
    print("  " + "-" * 65)
    print(f"  {'Total':<30} | {'':<20} | {total:>10,}")
    return total


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  [{func.__name__}] completed in {elapsed:.1f}s")
        return result
    return wrapper
