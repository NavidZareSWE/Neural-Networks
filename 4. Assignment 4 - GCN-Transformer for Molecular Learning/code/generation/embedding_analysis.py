import torch
import torch.nn.functional as F
from rdkit import Chem

import config
from data.featurizer import mol_to_graph
from data.vocab import encode
from torch_geometric.data import Batch
import selfies as sf


@torch.no_grad()
def build_training_combined(gcn, transformer, loader, device):
    gcn.eval(); transformer.eval()
    combined = []
    for graphs, tokens, pad_mask, eos_positions in loader:
        graphs = graphs.to(device)
        tokens, pad_mask, eos_positions = tokens.to(device), pad_mask.to(device), eos_positions.to(device)
        graph_embed = gcn(graphs)
        sequence_embed = transformer.sequence_representation(tokens, pad_mask, eos_positions)
        combined.append(F.normalize(torch.cat([graph_embed, sequence_embed], dim=-1), dim=-1).cpu())
    return torch.cat(combined, 0)


# ---- Generated-Sequence Representation ----
def generated_sequence_position(token_ids):
    ids = [int(t) for t in token_ids]
    if config.EOS_ID in ids:
        end = ids.index(config.EOS_ID)
        return ids[: end + 1], end, False
    return ids, len(ids) - 1, True


@torch.no_grad()
def _embed_molecule(mol, featurizer, token_to_id, gcn, transformer, device, token_ids=None):
    graph = mol_to_graph(mol, featurizer, target=0.0)
    graph_batch = Batch.from_data_list([graph]).to(device)
    graph_embed = gcn(graph_batch)

    if token_ids is not None:
        ids, eos_index, used_fallback = generated_sequence_position(token_ids)
    else:
        selfies_string = sf.encoder(Chem.MolToSmiles(mol))
        ids = encode(selfies_string, token_to_id, add_specials=True)
        eos_index = len(ids) - 1
        used_fallback = False

    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    pad_mask = torch.zeros_like(tokens, dtype=torch.bool)
    eos_position = torch.tensor([eos_index], dtype=torch.long, device=device)
    sequence_embed = transformer.sequence_representation(tokens, pad_mask, eos_position)

    combined = F.normalize(torch.cat([graph_embed, sequence_embed], dim=-1), dim=-1)
    graph_sequence_similarity = float((graph_embed * sequence_embed).sum().item())
    return combined.squeeze(0).cpu(), graph_sequence_similarity, used_fallback


def analyze_novel_molecules(novel_entries, gcn, transformer, featurizer, token_to_id,
                            training_combined, training_smiles, device=None, limit=None):
    device = device or config.DEVICE
    limit = limit or config.EMBED_NN_EXAMPLES
    examples = []
    fallback_count = 0
    for entry in novel_entries:
        try:
            combined, graph_sequence_similarity, used_fallback = _embed_molecule(
                entry["mol"], featurizer, token_to_id, gcn, transformer, device,
                token_ids=entry.get("token_ids"))
        except Exception:
            continue
        if used_fallback:
            fallback_count += 1
        similarities = training_combined @ combined
        nearest_index = int(torch.argmax(similarities).item())
        examples.append({
            "smiles": entry["smiles"],
            "temperature": entry.get("temperature"),
            "graph_sequence_similarity": round(graph_sequence_similarity, 4),
            "nearest_training_molecule": training_smiles[nearest_index],
            "nearest_embedding_similarity": round(float(similarities[nearest_index].item()), 4),
            "max_fingerprint_similarity": entry["max_fingerprint_similarity"],
            "eos_fallback_used": bool(used_fallback),
        })
        if len(examples) >= limit:
            break
    return examples, fallback_count
