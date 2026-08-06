import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw

import config


def plot_selfies_length_distribution(lengths, out_path=None, max_len=None):
    out_path = out_path or config.SELFIES_LEN_PLOT
    max_len = config.MAX_SELFIES_LEN if max_len is None else max_len
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(lengths, bins=range(0, max_len + 2, 2),
              color="#3b6ea5", edgecolor="white", linewidth=0.4)
    axis.set_xlabel("SELFIES sequence length (number of symbols)")
    axis.set_ylabel("Number of molecules")
    axis.set_title("SELFIES Sequence-Length Distribution")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout(); figure.savefig(out_path, dpi=150); plt.close(figure)
    return out_path


def plot_lm_curves(history, out_path=None):
    out_path = out_path or config.LM_CURVE_PLOT
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("LM warm-up token loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Token accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[2].plot(epochs, history["val_ppl"], color="#a5533b")
    axes[2].set_title("Validation perplexity"); axes[2].set_xlabel("epoch")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout(); figure.savefig(out_path, dpi=150); plt.close(figure)
    return out_path


def plot_joint_curves(history, out_path=None):
    out_path = out_path or config.JOINT_CURVE_PLOT
    epochs = range(1, len(history["train_total"]) + 1)
    val_total = [lm + config.CONTRASTIVE_WEIGHT * con + config.PAIR_WEIGHT * pair
                 for lm, con, pair in zip(history["val_lm"], history["val_contrastive"],
                                          history["val_pair"])]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epochs, history["train_total"], label="total (train)", color="black", linewidth=2)
    axes[0].plot(epochs, val_total, label="total (val)", color="gray", linestyle="--", linewidth=2)
    axes[0].plot(epochs, history["train_lm"], label="token (train)")
    axes[0].plot(epochs, history["val_lm"], label="token (val)")
    axes[0].plot(epochs, history["train_contrastive"], label="contrastive (train)")
    axes[0].plot(epochs, history["val_contrastive"], label="contrastive (val)")
    axes[0].plot(epochs, history["train_pair"], label="positive-pair (train)")
    axes[0].plot(epochs, history["val_pair"], label="positive-pair (val)")
    axes[0].set_title("Joint training losses"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=7)
    axes[1].plot(epochs, history["val_tok_acc"], label="token acc")
    axes[1].plot(epochs, history["val_retrieval_acc"], label="in-batch retrieval acc")
    axes[1].set_title("Validation accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[2].plot(epochs, history["val_ppl"], color="#a5533b")
    axes[2].set_title("Validation perplexity"); axes[2].set_xlabel("epoch")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout(); figure.savefig(out_path, dpi=150); plt.close(figure)
    return out_path


def plot_generation_metrics(per_temperature, out_path=None):
    out_path = out_path or config.GEN_METRICS_PLOT
    temperatures = sorted(per_temperature.keys())
    metric_names = ["rdkit_validity", "acceptance_rate", "uniqueness",
                    "exact_novelty", "structural_novelty"]
    positions = range(len(temperatures))
    width = 0.16
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for index, name in enumerate(metric_names):
        values = [per_temperature[t][name] for t in temperatures]
        axis.bar([p + index * width for p in positions], values, width, label=name)
    axis.set_xticks([p + 2 * width for p in positions])
    axis.set_xticklabels([f"T={t}" for t in temperatures])
    axis.set_ylim(0, 1.05); axis.set_ylabel("fraction")
    axis.set_title("Generation metrics by sampling temperature")
    axis.legend(fontsize=8); axis.grid(axis="y", alpha=0.3)
    figure.tight_layout(); figure.savefig(out_path, dpi=150); plt.close(figure)
    return out_path


def draw_molecule_grid(smiles_list, out_path=None, legends=None, mols_per_row=5, n=None):
    out_path = out_path or config.MOLECULE_GRID_PLOT
    n = n or config.GRID_SIZE
    smiles_list = smiles_list[:n]
    molecules, legend_labels = [], []
    for index, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        molecules.append(mol)
        legend_labels.append(legends[index] if legends else smiles)
    if not molecules:
        return None
    image = Draw.MolsToGridImage(molecules, molsPerRow=mols_per_row,
                                 subImgSize=(260, 220), legends=legend_labels)
    image.save(out_path)
    return out_path
