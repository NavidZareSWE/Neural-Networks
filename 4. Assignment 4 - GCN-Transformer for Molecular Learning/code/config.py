import os
from pathlib import Path

import torch as _torch

SEED = 42

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CSV = str(PROJECT_ROOT / "Lipophilicity.csv")
RAW_CSV = os.environ.get("LIPO_CSV", _DEFAULT_CSV)

OUTPUT_DIR = str(PROJECT_ROOT / "output")
OUTPUT_SECTION1 = os.path.join(OUTPUT_DIR, "section1")
OUTPUT_SECTION2 = os.path.join(OUTPUT_DIR, "section2")
for _d in (OUTPUT_DIR, OUTPUT_SECTION1, OUTPUT_SECTION2):
    os.makedirs(_d, exist_ok=True)

DATASET_SPLITS_JSON = os.path.join(OUTPUT_SECTION1, "dataset_splits.json")
VOCAB_JSON = os.path.join(OUTPUT_SECTION1, "vocab.json")
STATS_JSON = os.path.join(OUTPUT_SECTION1, "stats.json")
SELFIES_LEN_PLOT = os.path.join(OUTPUT_SECTION1, "selfies_length_distribution.png")

LM_CURVE_PLOT = os.path.join(OUTPUT_SECTION2, "lm_pretrain_curves.png")
JOINT_CURVE_PLOT = os.path.join(OUTPUT_SECTION2, "joint_training_curves.png")
GEN_METRICS_PLOT = os.path.join(OUTPUT_SECTION2, "generation_metrics.png")
MOLECULE_GRID_PLOT = os.path.join(OUTPUT_SECTION2, "accepted_novel_grid.png")
RESULTS_JSON = os.path.join(OUTPUT_SECTION2, "results.json")
TRAINING_LOG = os.path.join(OUTPUT_SECTION2, "training_log.txt")
CHECKPOINT = os.path.join(OUTPUT_SECTION2, "checkpoint.pt")
REJECTED_JSON = os.path.join(OUTPUT_SECTION2, "rejected_examples.json")

# ---- Dataset Preparation ----
MAX_SELFIES_LEN = 100
SPLIT_FRACTIONS = (0.80, 0.10, 0.10)

# ---- Special Tokens ----
PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

# ---- Bond-Order Edge Weights ----
BOND_WEIGHTS = {1.0: 1.0, 2.0: 2.0, 3.0: 3.0, 1.5: 1.5}
DEFAULT_BOND_WEIGHT = 1.0

# ---- GCN Encoder ----
GCN_HIDDEN = 192
GCN_LAYERS = 3
GCN_DROPOUT = 0.10
ATOM_EMB_DIM = GCN_HIDDEN

# ---- Transformer ----
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 4
DIM_FF = 512
TF_DROPOUT = 0.10
MAX_POSITIONS = MAX_SELFIES_LEN + 10

# ---- Shared Contrastive Space ----
PROJ_DIM = 128
CONTRASTIVE_TAU = 0.07
CONTRASTIVE_WEIGHT = 1.0
PAIR_WEIGHT = 0.25

# ---- Optimisation ----
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
LR = 3e-4
WEIGHT_DECAY = 1e-5
LM_EPOCHS = int(os.environ.get("LM_EPOCHS", 15))
JOINT_EPOCHS = int(os.environ.get("JOINT_EPOCHS", 40))
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", 8))

# ---- Generation ----
TEMPERATURES = [0.8, 1.0, 1.2]
N_GENERATE = int(os.environ.get("N_GENERATE", 300))
MAX_GEN_LEN = MAX_SELFIES_LEN
TOP_K = 20
TOP_P = 0.95
EOS_MIN_STEP = 5

# ---- Validation / Novelty ----
FP_RADIUS = 2
FP_BITS = 2048
STRUCT_NOVELTY_TANIMOTO = 0.85
GRID_SIZE = 20
EMBED_NN_EXAMPLES = 12
MAX_REJECTED_PER_STAGE = 5

# ---- Device / Smoke ----
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

SMOKE = os.environ.get("SMOKE", "0") == "1"
if SMOKE:
    SMOKE_SUBSET = 96
    LM_EPOCHS = 1
    JOINT_EPOCHS = 2
    BATCH_SIZE = 16
    N_GENERATE = 40
    EARLY_STOP_PATIENCE = 5
