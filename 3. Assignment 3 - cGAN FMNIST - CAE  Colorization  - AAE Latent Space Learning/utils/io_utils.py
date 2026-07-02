import numpy as np
import os
import json


def ensure_dirs(base_dir='output'):
    dirs = [
        os.path.join(base_dir, 'training'),
        os.path.join(base_dir, 'evaluation'),
        os.path.join(base_dir, 'plots'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def save_metrics(metrics, filepath):
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(filepath, 'w') as f:
        json.dump(convert(metrics), f, indent=2)
    print(f"    [SAVED] Metrics -> {filepath}")


def save_training_history(history, filepath):
    save_dict = {}
    for key, val in history.items():
        if key == 'samples':
            continue
        save_dict[key] = np.array(val)
    np.savez(filepath, **save_dict)
    print(f"    [SAVED] Training history -> {filepath}")
