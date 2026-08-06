import torch

import config


@torch.no_grad()
def encode_split(gcn, transformer, loader, device):
    gcn.eval(); transformer.eval()
    graph_embeddings, sequence_embeddings = [], []
    for graphs, tokens, pad_mask, eos_positions in loader:
        graphs = graphs.to(device)
        tokens, pad_mask, eos_positions = tokens.to(device), pad_mask.to(device), eos_positions.to(device)
        graph_embeddings.append(gcn(graphs).cpu())
        sequence_embeddings.append(
            transformer.sequence_representation(tokens, pad_mask, eos_positions).cpu())
    return torch.cat(graph_embeddings, 0), torch.cat(sequence_embeddings, 0)


def _directional_metrics(similarity):
    n = similarity.size(0)
    ranks = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        order = torch.argsort(similarity[i], descending=True)
        ranks[i] = (order == i).nonzero(as_tuple=True)[0].item()
    ranks = ranks + 1
    return {
        "top_1": float((ranks <= 1).float().mean().item()),
        "top_5": float((ranks <= 5).float().mean().item()),
        "top_10": float((ranks <= 10).float().mean().item()),
        "mean_rank": float(ranks.float().mean().item()),
    }


def evaluate_retrieval(gcn, transformer, loader, device=None):
    device = device or config.DEVICE
    graph_embed, sequence_embed = encode_split(gcn, transformer, loader, device)
    similarity = graph_embed @ sequence_embed.t()
    n = similarity.size(0)

    diagonal = similarity.diag()
    off_diagonal_sum = similarity.sum() - diagonal.sum()
    matching_mean = float(diagonal.mean().item())
    nonmatching_mean = float((off_diagonal_sum / (n * n - n)).item())

    return {
        "graph_to_sequence": _directional_metrics(similarity),
        "sequence_to_graph": _directional_metrics(similarity.t()),
        "matching_pair_cosine": matching_mean,
        "nonmatching_pair_cosine": nonmatching_mean,
        "random_chance_top_1": float(1.0 / n),
        "random_chance_top_5": float(min(5.0 / n, 1.0)),
        "random_chance_top_10": float(min(10.0 / n, 1.0)),
        "n": int(n),
    }
