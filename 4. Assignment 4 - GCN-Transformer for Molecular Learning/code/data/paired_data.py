import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from rdkit import Chem

import config
from data.featurizer import mol_to_graph
from data.vocab import encode


class PairedMoleculeDataset(Dataset):
    def __init__(self, records, featurizer, token_to_id):
        self.samples = []
        for record in records:
            mol = Chem.MolFromSmiles(record["smiles"])
            if mol is None:
                continue
            graph = mol_to_graph(mol, featurizer, target=record["exp"])
            token_ids = encode(record["selfies"], token_to_id, add_specials=True)
            self.samples.append((graph, torch.tensor(token_ids, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate(batch):
    graphs = [graph for graph, _ in batch]
    sequences = [sequence for _, sequence in batch]
    graph_batch = Batch.from_data_list(graphs)

    lengths = torch.tensor([sequence.size(0) for sequence in sequences], dtype=torch.long)
    max_length = int(lengths.max().item())
    tokens = torch.full((len(sequences), max_length), config.PAD_ID, dtype=torch.long)
    for i, sequence in enumerate(sequences):
        tokens[i, : sequence.size(0)] = sequence

    pad_mask = tokens.eq(config.PAD_ID)
    eos_positions = lengths - 1
    return graph_batch, tokens, pad_mask, eos_positions


def make_loader(records, featurizer, token_to_id, batch_size=None, shuffle=True):
    batch_size = batch_size or config.BATCH_SIZE
    dataset = PairedMoleculeDataset(records, featurizer, token_to_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
