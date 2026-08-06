import numpy as np
import selfies as sf
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator

import config

_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=config.FP_RADIUS, fpSize=config.FP_BITS)


def _fingerprint(mol):
    return _FP_GENERATOR.GetFingerprint(mol)


# ---- Training Reference ----
def build_training_reference(train_records):
    canonical_set = set()
    fingerprints = []
    elements = set()
    heavy_atoms = []
    molecular_weights = []
    max_abs_charge = 0
    for record in train_records:
        mol = Chem.MolFromSmiles(record["smiles"])
        if mol is None:
            continue
        canonical_set.add(Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True))
        fingerprints.append(_fingerprint(mol))
        for atom in mol.GetAtoms():
            elements.add(atom.GetAtomicNum())
            max_abs_charge = max(max_abs_charge, abs(atom.GetFormalCharge()))
        heavy_atoms.append(mol.GetNumHeavyAtoms())
        molecular_weights.append(Descriptors.MolWt(mol))
    return {
        "canonical_set": canonical_set,
        "fingerprints": fingerprints,
        "elements": elements,
        "heavy_low": float(np.percentile(heavy_atoms, 1)),
        "heavy_high": float(np.percentile(heavy_atoms, 99)),
        "weight_low": float(np.percentile(molecular_weights, 1)),
        "weight_high": float(np.percentile(molecular_weights, 99)),
        "max_abs_charge": int(max_abs_charge),
    }


# ---- Per-Sample Validation ----
def _passes_pipeline(selfies_string, reference):
    if not selfies_string:
        return "empty", None, None
    try:
        smiles = sf.decoder(selfies_string)
    except Exception:
        return "selfies_decode_fail", None, None
    if not smiles:
        return "selfies_decode_fail", None, None

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return "parse_none", None, smiles
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return "sanitize_fail", None, smiles

    if len(Chem.GetMolFrags(mol)) != 1:
        return "multi_fragment", "rdkit_valid", smiles

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 or atom.GetAtomicNum() not in reference["elements"]:
            return "foreign_element", "rdkit_valid", smiles
    if any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms()):
        return "radical", "rdkit_valid", smiles

    heavy = mol.GetNumHeavyAtoms()
    weight = Descriptors.MolWt(mol)
    if not (reference["heavy_low"] <= heavy <= reference["heavy_high"]):
        return "out_of_range", "rdkit_valid", smiles
    if not (reference["weight_low"] <= weight <= reference["weight_high"]):
        return "out_of_range", "rdkit_valid", smiles

    max_charge = max((abs(atom.GetFormalCharge()) for atom in mol.GetAtoms()), default=0)
    if max_charge > reference["max_abs_charge"]:
        return "formal_charge", "rdkit_valid", smiles

    return "accepted", "rdkit_valid", smiles


# ---- Batch Validation + Metrics ----
def validate_generations(selfies_strings, reference):
    stages = ["empty", "selfies_decode_fail", "parse_none", "sanitize_fail",
              "multi_fragment", "foreign_element", "radical", "out_of_range",
              "formal_charge", "accepted"]
    counts = {stage: 0 for stage in stages}
    counts["total"] = len(selfies_strings)
    rdkit_valid = 0

    accepted_canonical = {}
    rejected_examples = {}
    max_rejected_per_stage = config.MAX_REJECTED_PER_STAGE
    for sample_index, selfies_string in enumerate(selfies_strings):
        outcome, validity_flag, smiles = _passes_pipeline(selfies_string, reference)
        counts[outcome] += 1
        if validity_flag == "rdkit_valid":
            rdkit_valid += 1
        if outcome == "accepted":
            mol = Chem.MolFromSmiles(sf.decoder(selfies_string))
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            accepted_canonical.setdefault(canonical, (mol, sample_index))
        else:
            bucket = rejected_examples.setdefault(outcome, [])
            if len(bucket) < max_rejected_per_stage:
                bucket.append({"stage": outcome, "sample_index": int(sample_index),
                               "selfies": selfies_string, "smiles": smiles})

    accepted_total = counts["accepted"]
    unique_molecules = list(accepted_canonical.items())

    novel_entries = []
    qed_values = []
    for canonical, (mol, sample_index) in unique_molecules:
        qed_values.append(QED.qed(mol))
        if canonical in reference["canonical_set"]:
            continue
        fingerprint = _fingerprint(mol)
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, reference["fingerprints"])
        max_similarity = max(similarities) if similarities else 0.0
        novel_entries.append({
            "smiles": canonical,
            "mol": mol,
            "sample_index": int(sample_index),
            "max_fingerprint_similarity": round(float(max_similarity), 4),
            "structurally_novel": bool(max_similarity < config.STRUCT_NOVELTY_TANIMOTO),
        })

    unique_count = max(len(unique_molecules), 1)
    metrics = {
        "rdkit_validity": rdkit_valid / counts["total"] if counts["total"] else 0.0,
        "acceptance_rate": accepted_total / counts["total"] if counts["total"] else 0.0,
        "uniqueness": len(unique_molecules) / max(accepted_total, 1),
        "exact_novelty": len(novel_entries) / unique_count,
        "structural_novelty": sum(e["structurally_novel"] for e in novel_entries) / unique_count,
        "mean_qed": float(np.mean(qed_values)) if qed_values else 0.0,
    }
    rejected_flat = [example for bucket in rejected_examples.values() for example in bucket]
    return counts, metrics, novel_entries, rejected_flat
