import torch
from torch_geometric.data import Data
from rdkit import Chem

import config

MAX_DEGREE = 6
MIN_CHARGE, MAX_CHARGE = -2, 2
MAX_HYDROGENS = 4
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]
CHIRALITIES = [
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.ChiralType.CHI_OTHER,
]


def bucket_index(value, low, high):
    if value < low or value > high:
        return high - low + 1
    return value - low


class AtomFeaturizer:
    def __init__(self):
        self.element_to_index = {}
        self.field_sizes = []
        self.n_fields = 0

    def fit(self, molecules):
        elements = set()
        for mol in molecules:
            for atom in mol.GetAtoms():
                elements.add(atom.GetAtomicNum())
        self.element_to_index = {atomic_number: i for i, atomic_number in enumerate(sorted(elements))}
        self.field_sizes = [
            len(self.element_to_index) + 1,
            (MAX_DEGREE + 1) + 1,
            (MAX_CHARGE - MIN_CHARGE + 1) + 1,
            len(HYBRIDIZATIONS) + 1,
            2,
            (MAX_HYDROGENS + 1) + 1,
            len(CHIRALITIES) + 1,
        ]
        self.n_fields = len(self.field_sizes)
        return self

    def atom_indices(self, atom):
        element_index = self.element_to_index.get(atom.GetAtomicNum(), len(self.element_to_index))
        degree_index = bucket_index(atom.GetDegree(), 0, MAX_DEGREE)
        charge_index = bucket_index(atom.GetFormalCharge(), MIN_CHARGE, MAX_CHARGE)
        hybridization = atom.GetHybridization()
        hybridization_index = (HYBRIDIZATIONS.index(hybridization)
                               if hybridization in HYBRIDIZATIONS else len(HYBRIDIZATIONS))
        aromatic_index = 1 if atom.GetIsAromatic() else 0
        hydrogen_index = bucket_index(atom.GetTotalNumHs(), 0, MAX_HYDROGENS)
        chirality = atom.GetChiralTag()
        chirality_index = (CHIRALITIES.index(chirality)
                           if chirality in CHIRALITIES else len(CHIRALITIES))
        return [element_index, degree_index, charge_index, hybridization_index,
                aromatic_index, hydrogen_index, chirality_index]


def mol_to_graph(mol, featurizer, target=0.0):
    node_features = torch.tensor([featurizer.atom_indices(atom) for atom in mol.GetAtoms()],
                                 dtype=torch.long)
    sources, destinations, weights = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        weight = config.BOND_WEIGHTS.get(bond.GetBondTypeAsDouble(), config.DEFAULT_BOND_WEIGHT)
        sources += [i, j]; destinations += [j, i]; weights += [weight, weight]

    if sources:
        edge_index = torch.tensor([sources, destinations], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)
    data.y = torch.tensor([target], dtype=torch.float)
    data.num_nodes = node_features.size(0)
    return data


def build_featurizer_from_smiles(smiles_list):
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    molecules = [mol for mol in molecules if mol is not None]
    return AtomFeaturizer().fit(molecules)
