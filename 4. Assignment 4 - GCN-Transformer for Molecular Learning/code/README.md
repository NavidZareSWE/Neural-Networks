# NNDL Assignment 4 — GCN + Transformer for Molecular Representation and Generation

Graph Convolutional Network (graph view) and a small causal Transformer (SELFIES
sequence view) trained on the MoleculeNet Lipophilicity dataset. The two views are
aligned with a contrastive objective; the Transformer is additionally trained with
next-token prediction and used to generate new molecules, which are validated with
RDKit.

## Project structure

```
nndl_project4/
├── config.py                 # all hyperparameters, paths, special-token ids, device
├── main.py                   # full pipeline: data -> models -> train -> generate -> validate
├── run_section1_data.py      # data stage only
│
├── data/
│   ├── dataset.py            # sanitize -> canonical isomeric SMILES -> dedup -> SELFIES -> length filter -> splits
│   ├── vocab.py              # train-only SELFIES vocabulary + encode/decode
│   ├── featurizer.py         # per-field atom featurizer + mol_to_graph (bond-order edge weights)
│   ├── paired_data.py        # (graph, token-sequence) dataset + collate (PyG Batch + padding)
│   └── prepare.py            # runs data stage, prints stats, writes section1 artifacts
│
├── models/
│   ├── gcn.py                # 3-layer GCNConv encoder + L2-normalized projection head
│   └── transformer.py        # causal Transformer (causal + padding masks) + generation
│
├── training/
│   ├── losses.py             # InfoNCE contrastive + next-token LM loss / accuracy / perplexity
│   ├── pretrain_lm.py        # stage 1: language-model warm-up
│   └── train_joint.py        # stage 2: joint LM + contrastive
│
├── generation/
│   ├── sampler.py            # temperature sampling -> SELFIES strings
│   ├── validate.py           # SELFIES decode + RDKit + novelty + fingerprint nearest-neighbor
│   └── retrieval.py          # bidirectional graph<->sequence retrieval metrics
│
├── visualize.py              # ALL plots: length distribution, curves, generation metrics, molecule grid
│
└── output/
    ├── section1/             # dataset_splits.json, vocab.json, stats.json, selfies_length_distribution.png
    └── section2/             # curves, generation metrics, accepted_novel_grid.png, results.json
```

## Requirements

```
pip install torch torch_geometric rdkit selfies matplotlib numpy
```

Domestic PyPI mirror (per the assignment):

```
pip install <pkg> --index-url https://package-mirror.liara.ir/repository/pypi/simple
```

## How to run

Data stage only (fast, CPU):

```
python3 run_section1_data.py
```

Full pipeline (GPU recommended — designed for ~30 min on a Kaggle GPU):

```
python3 main.py
```

To run on Kaggle, see `KAGGLE.md` for step-by-step instructions, or import
`run_on_kaggle.ipynb` directly as a Kaggle notebook.

Fast dry run to verify the pipeline end-to-end on CPU (tiny subset, 1 epoch each):

```
SMOKE=1 python3 main.py
```

All hyperparameters live in `config.py`. `SMOKE=1` shrinks the dataset, epochs,
batch size, and generation count so the whole pipeline can be exercised in a few
minutes on CPU; results from a smoke run are for verification only and are not
representative.

## Outputs

Section 1 (`output/section1/`): dataset splits, vocabulary, statistics, and the
SELFIES length-distribution plot.

Section 2 (`output/section2/`): LM warm-up curves, joint-training curves,
generation metrics by temperature, the grid of accepted novel molecules, a
consolidated `results.json` (dataset stats, retrieval metrics, per-temperature
generation counts/metrics, and accepted-molecule examples), and `training_log.txt`.

## Provenance of the shipped `output/` artifacts

The `output/section2/` folder is populated by running `python3 main.py`. The default
settings in `config.py` follow the assignment: 15 language-model warm-up epochs, up to
40 joint epochs with validation-based early stopping, batch size 256, mixed precision,
cosine learning-rate decay, and 300 generated sequences at each temperature
(0.8, 1.0, 1.2). On a Kaggle GPU the full run completes in a few minutes.
