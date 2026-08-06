import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import random

import numpy as np
import torch

import config
import visualize
from data.prepare import prepare_and_report
from data.featurizer import build_featurizer_from_smiles
from data.paired_data import make_loader
from models.gcn import GCNEncoder
from models.transformer import CausalTransformer
from training.pretrain_lm import pretrain_language_model
from training.train_joint import train_joint
from generation.sampler import sample_molecules
from generation.validate import build_training_reference, validate_generations
from generation.retrieval import evaluate_retrieval
from generation.embedding_analysis import build_training_combined, analyze_novel_molecules


def set_seed(seed=config.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def section(title):
    print("\n" + "#" * 68)
    print(title)
    print("#" * 68)


def write_training_log(lm_history, joint_history, retrieval, generation_report,
                       n_novel, nn_examples):
    lines = ["LANGUAGE-MODEL WARM-UP"]
    for i in range(len(lm_history["train_loss"])):
        lines.append(f"[LM {i + 1:02d}] train_loss={lm_history['train_loss'][i]:.4f} "
                     f"val_loss={lm_history['val_loss'][i]:.4f} "
                     f"val_tok_acc={lm_history['val_acc'][i]:.4f} "
                     f"val_ppl={lm_history['val_ppl'][i]:.3f}")
    lines.append("")
    lines.append("JOINT TRAINING")
    for i in range(len(joint_history["train_total"])):
        lines.append(f"[joint {i + 1:02d}] total={joint_history['train_total'][i]:.4f} "
                     f"lm={joint_history['train_lm'][i]:.4f} "
                     f"con={joint_history['train_contrastive'][i]:.4f} "
                     f"pair={joint_history['train_pair'][i]:.4f} "
                     f"val_tok_acc={joint_history['val_tok_acc'][i]:.4f} "
                     f"val_ret_acc={joint_history['val_retrieval_acc'][i]:.4f} "
                     f"val_ppl={joint_history['val_ppl'][i]:.3f}")
    lines.append("")
    lines.append("RETRIEVAL")
    lines.append(f"graph->sequence: {retrieval['graph_to_sequence']}")
    lines.append(f"sequence->graph: {retrieval['sequence_to_graph']}")
    lines.append(f"matching_cosine={retrieval['matching_pair_cosine']:.4f} "
                 f"nonmatching_cosine={retrieval['nonmatching_pair_cosine']:.4f}")
    lines.append("")
    lines.append("GENERATION")
    for temperature, report in generation_report.items():
        lines.append(f"T={temperature} counts={report['counts']}")
        lines.append(f"T={temperature} metrics={report['metrics']}")
    lines.append("")
    lines.append(f"Unique accepted novel molecules: {n_novel}")
    with open(config.TRAINING_LOG, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    set_seed()
    device = config.DEVICE
    print(f"Device: {device} | SMOKE={config.SMOKE}")

    # ---- Section 1: Data ----
    splits, stats, token_to_id, id_to_token = prepare_and_report()
    if config.SMOKE:
        for key in splits:
            splits[key] = splits[key][: config.SMOKE_SUBSET]
    vocab_size = len(token_to_id)

    training_smiles = [record["smiles"] for record in splits["train"]]
    featurizer = build_featurizer_from_smiles(training_smiles)

    train_loader = make_loader(splits["train"], featurizer, token_to_id, shuffle=True)
    train_reference_loader = make_loader(splits["train"], featurizer, token_to_id, shuffle=False)
    val_loader = make_loader(splits["val"], featurizer, token_to_id, shuffle=False)
    test_loader = make_loader(splits["test"], featurizer, token_to_id, shuffle=False)

    # ---- Models ----
    gcn = GCNEncoder(featurizer.field_sizes).to(device)
    transformer = CausalTransformer(vocab_size).to(device)
    print(f"GCN parameters        : {sum(p.numel() for p in gcn.parameters()):,}")
    print(f"Transformer parameters: {sum(p.numel() for p in transformer.parameters()):,}")

    # ---- Stage 1: Language-Model Warm-Up ----
    section("STAGE 1 -- LANGUAGE-MODEL WARM-UP")
    lm_history = pretrain_language_model(transformer, train_loader, val_loader, device)
    visualize.plot_lm_curves(lm_history)

    # ---- Stage 2: Joint Contrastive Training ----
    section("STAGE 2 -- JOINT CONTRASTIVE TRAINING")
    joint_history = train_joint(gcn, transformer, train_loader, val_loader, device)
    visualize.plot_joint_curves(joint_history)

    # ---- Retrieval ----
    section("RETRIEVAL (TEST SPLIT)")
    retrieval = evaluate_retrieval(gcn, transformer, test_loader, device)
    print("graph -> sequence:", retrieval["graph_to_sequence"])
    print("sequence -> graph:", retrieval["sequence_to_graph"])
    print(f"matching-pair cosine    : {retrieval['matching_pair_cosine']:.4f}")
    print(f"nonmatching-pair cosine : {retrieval['nonmatching_pair_cosine']:.4f}")
    print(f"random-chance top-1/top-5/top-10: {retrieval['random_chance_top_1']:.4f} / "
          f"{retrieval['random_chance_top_5']:.4f} / {retrieval['random_chance_top_10']:.4f}")

    # ---- Generation + Validation ----
    section("GENERATION + VALIDATION")
    reference = build_training_reference(splits["train"])
    training_combined = build_training_combined(gcn, transformer, train_reference_loader, device)

    generation_report = {}
    per_temperature = {}
    all_novel = []
    for temperature in config.TEMPERATURES:
        selfies_strings, token_sequences = sample_molecules(
            transformer, id_to_token, n=config.N_GENERATE,
            temperature=temperature, device=device)
        counts, metrics, novel_entries, rejected = validate_generations(selfies_strings, reference)
        for entry in novel_entries:
            entry["token_ids"] = token_sequences[entry["sample_index"]]
            entry["temperature"] = float(temperature)
        per_temperature[temperature] = metrics
        generation_report[str(temperature)] = {"counts": counts, "metrics": metrics,
                                                "rejected_examples": rejected}
        all_novel.extend(novel_entries)
        print(f"\nTemperature {temperature}:")
        print(f"  counts : {counts}")
        print("  metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if rejected:
            print("  rejected examples:")
            for example in rejected:
                print(f"    [{example['stage']}] {example['smiles']}")

    visualize.plot_generation_metrics(per_temperature)

    # ---- Embedding Nearest-Neighbor Analysis ----
    section("EMBEDDING NEAREST-NEIGHBOR ANALYSIS (ACCEPTED NOVEL)")
    unique_novel = {}
    for entry in all_novel:
        unique_novel.setdefault(entry["smiles"], entry)
    novel_list = sorted(unique_novel.values(), key=lambda e: e["max_fingerprint_similarity"])
    nn_examples, eos_fallback_count = analyze_novel_molecules(
        novel_list, gcn, transformer, featurizer, token_to_id,
        training_combined, training_smiles, device)
    print(f"  sequences using the final-token fallback (no <eos> emitted): {eos_fallback_count}"
          f" of {len(nn_examples)}")
    for example in nn_examples:
        print(f"  gen={example['smiles']}")
        print(f"     graph-seq sim={example['graph_sequence_similarity']:.4f} | "
              f"nearest train sim={example['nearest_embedding_similarity']:.4f} | "
              f"max fp sim={example['max_fingerprint_similarity']:.4f}")
        print(f"     nearest training molecule={example['nearest_training_molecule']}")

    # ---- Accepted Novel Grid ----
    grid_smiles = [entry["smiles"] for entry in novel_list[: config.GRID_SIZE]]
    grid_legends = [f"fp sim={entry['max_fingerprint_similarity']:.2f}"
                    for entry in novel_list[: config.GRID_SIZE]]
    grid_path = visualize.draw_molecule_grid(grid_smiles, legends=grid_legends)
    print(f"\nAccepted novel molecules (unique): {len(novel_list)}")
    print(f"Grid image: {grid_path}")

    # ---- Persist Results ----
    results = {
        "dataset_stats": stats,
        "lm_history": lm_history,
        "joint_history": joint_history,
        "final_token_accuracy": joint_history["val_tok_acc"][-1],
        "final_perplexity": joint_history["val_ppl"][-1],
        "retrieval": retrieval,
        "generation": generation_report,
        "n_accepted_novel_unique": len(novel_list),
        "embedding_nn_examples": nn_examples,
        "grid_examples": [
            {"smiles": e["smiles"],
             "max_fingerprint_similarity": e["max_fingerprint_similarity"],
             "structurally_novel": e["structurally_novel"]}
            for e in novel_list[: config.GRID_SIZE]
        ],
    }
    results["eos_fallback_count"] = int(eos_fallback_count)
    with open(config.RESULTS_JSON, "w") as handle:
        json.dump(results, handle, indent=2)

    rejected_dump = {temperature: report["rejected_examples"]
                     for temperature, report in generation_report.items()}
    with open(config.REJECTED_JSON, "w") as handle:
        json.dump(rejected_dump, handle, indent=2)

    torch.save({
        "gcn_state": gcn.state_dict(),
        "transformer_state": transformer.state_dict(),
        "token_to_id": token_to_id,
        "field_sizes": featurizer.field_sizes,
        "element_to_index": featurizer.element_to_index,
        "vocab_size": vocab_size,
    }, config.CHECKPOINT)
    print(f"Checkpoint written: {config.CHECKPOINT}")
    print(f"Rejected examples written: {config.REJECTED_JSON}")
    write_training_log(lm_history, joint_history, retrieval, generation_report,
                       len(novel_list), nn_examples)
    print(f"\nResults written: {config.RESULTS_JSON}")
    print(f"Training log written: {config.TRAINING_LOG}")


if __name__ == "__main__":
    main()
