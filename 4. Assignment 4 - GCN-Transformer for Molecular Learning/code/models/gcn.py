import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

import config


class GCNEncoder(nn.Module):
    def __init__(self, field_sizes, hidden=None, layers=None, emb_dim=None,
                 dropout=None, proj_dim=None):
        super().__init__()
        hidden = hidden or config.GCN_HIDDEN
        layers = layers or config.GCN_LAYERS
        emb_dim = emb_dim or config.ATOM_EMB_DIM
        dropout = config.GCN_DROPOUT if dropout is None else dropout
        proj_dim = proj_dim or config.PROJ_DIM

        self.field_embeddings = nn.ModuleList(
            [nn.Embedding(size, emb_dim) for size in field_sizes])
        self.input_projection = nn.Linear(emb_dim, hidden) if emb_dim != hidden else nn.Identity()

        self.convolutions = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(layers)])
        self.normalizations = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.dropout = dropout

        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim),
        )

    def pool_graph(self, x, edge_index, edge_weight, batch):
        h = sum(embedding(x[:, i]) for i, embedding in enumerate(self.field_embeddings))
        h = self.input_projection(h)
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            residual = h
            h = convolution(h, edge_index, edge_weight)
            h = F.relu(h)
            h = normalization(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + residual
        return global_mean_pool(h, batch)

    def forward(self, data):
        pooled = self.pool_graph(data.x, data.edge_index, data.edge_weight, data.batch)
        return F.normalize(self.projection(pooled), dim=-1)
