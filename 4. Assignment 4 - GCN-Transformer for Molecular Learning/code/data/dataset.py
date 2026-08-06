import csv
import random

from rdkit import Chem
from rdkit import RDLogger
import selfies as sf

import config

RDLogger.DisableLog("rdApp.*")


def read_raw_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((row["smiles"], float(row["exp"]), row["CMPD_CHEMBLID"]))
    return rows


def canonical_isomeric(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    canonical_mol = Chem.MolFromSmiles(canonical)
    if canonical_mol is None:
        return None, None
    return canonical, canonical_mol


def encode_selfies(canonical_smiles):
    try:
        encoded = sf.encoder(canonical_smiles)
    except Exception:
        return None
    return encoded or None


def build_dataset(csv_path=None, max_len=None, split_fractions=None, seed=None):
    csv_path = csv_path or config.RAW_CSV
    max_len = config.MAX_SELFIES_LEN if max_len is None else max_len
    split_fractions = split_fractions or config.SPLIT_FRACTIONS
    seed = config.SEED if seed is None else seed

    raw_rows = read_raw_rows(csv_path)
    n_original = len(raw_rows)

    n_failed_parse = n_failed_selfies = n_too_long = n_duplicates = 0
    seen_canonical = set()
    records = []

    for smiles, exp, chembl_id in raw_rows:
        canonical, canonical_mol = canonical_isomeric(smiles)
        if canonical is None:
            n_failed_parse += 1
            continue
        if canonical in seen_canonical:
            n_duplicates += 1
            continue
        seen_canonical.add(canonical)

        selfies_string = encode_selfies(canonical)
        if selfies_string is None:
            n_failed_selfies += 1
            continue
        selfies_length = sf.len_selfies(selfies_string)
        if selfies_length > max_len:
            n_too_long += 1
            continue

        records.append({
            "smiles": canonical,
            "selfies": selfies_string,
            "exp": exp,
            "chembl_id": chembl_id,
            "n_atoms": canonical_mol.GetNumAtoms(),
            "n_bonds": canonical_mol.GetNumBonds(),
            "sel_len": selfies_length,
        })

    n_usable = len(records)

    # ---- Fixed Splits ----
    generator = random.Random(seed)
    indices = list(range(n_usable))
    generator.shuffle(indices)
    train_fraction, val_fraction, _ = split_fractions
    n_train = int(round(train_fraction * n_usable))
    n_val = int(round(val_fraction * n_usable))
    train_indices = set(indices[:n_train])
    val_indices = set(indices[n_train:n_train + n_val])

    splits = {"train": [], "val": [], "test": []}
    for position, record in enumerate(records):
        if position in train_indices:
            splits["train"].append(record)
        elif position in val_indices:
            splits["val"].append(record)
        else:
            splits["test"].append(record)

    stats = {
        "n_original": n_original,
        "n_failed_parse": n_failed_parse,
        "n_failed_selfies": n_failed_selfies,
        "n_too_long": n_too_long,
        "n_removed_total": n_failed_parse + n_failed_selfies + n_too_long,
        "n_duplicates": n_duplicates,
        "n_usable": n_usable,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "avg_atoms": sum(r["n_atoms"] for r in records) / n_usable,
        "avg_bonds": sum(r["n_bonds"] for r in records) / n_usable,
        "sel_len_min": min(r["sel_len"] for r in records),
        "sel_len_avg": sum(r["sel_len"] for r in records) / n_usable,
        "sel_len_max": max(r["sel_len"] for r in records),
    }
    return splits, stats
