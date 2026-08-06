import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def build_causal_mask(size, device):
    mask = torch.full((size, size), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def filter_logits(logits, top_k, top_p):
    filtered = logits.clone()
    if top_k and top_k > 0:
        k = min(top_k, filtered.size(-1))
        kth_value = torch.topk(filtered, k, dim=-1).values[:, -1, None]
        filtered[filtered < kth_value] = float("-inf")
    if top_p and top_p < 1.0:
        sorted_logits, sorted_index = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        drop = cumulative > top_p
        drop[:, 1:] = drop[:, :-1].clone()
        drop[:, 0] = False
        sorted_logits[drop] = float("-inf")
        filtered = torch.full_like(filtered, float("-inf")).scatter(-1, sorted_index, sorted_logits)
    return filtered


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=None, n_heads=None, n_layers=None,
                 dim_ff=None, dropout=None, max_positions=None, proj_dim=None):
        super().__init__()
        d_model = d_model or config.D_MODEL
        n_heads = n_heads or config.N_HEADS
        n_layers = n_layers or config.N_LAYERS
        dim_ff = dim_ff or config.DIM_FF
        dropout = config.TF_DROPOUT if dropout is None else dropout
        max_positions = max_positions or config.MAX_POSITIONS
        proj_dim = proj_dim or config.PROJ_DIM

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=config.PAD_ID)
        self.position_embedding = nn.Embedding(max_positions, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim),
        )
        self.max_positions = max_positions

    def hidden_states(self, tokens, pad_mask):
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device).unsqueeze(0).expand(batch, length)
        x = self.embedding_dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        causal = build_causal_mask(length, tokens.device)
        return self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)

    def forward(self, tokens, pad_mask):
        hidden = self.hidden_states(tokens, pad_mask)
        return self.lm_head(hidden), hidden

    def project_eos(self, hidden, eos_positions):
        gather_index = eos_positions.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        eos_hidden = hidden.gather(1, gather_index).squeeze(1)
        return F.normalize(self.projection(eos_hidden), dim=-1)

    def sequence_representation(self, tokens, pad_mask, eos_positions):
        hidden = self.hidden_states(tokens, pad_mask)
        return self.project_eos(hidden, eos_positions)

    @torch.no_grad()
    def generate(self, n, temperature=1.0, top_k=None, top_p=None,
                 max_len=None, device=None):
        top_k = config.TOP_K if top_k is None else top_k
        top_p = config.TOP_P if top_p is None else top_p
        max_len = max_len or config.MAX_GEN_LEN
        device = device or next(self.parameters()).device
        self.eval()

        tokens = torch.full((n, 1), config.BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        blocked = [config.PAD_ID, config.BOS_ID, config.UNK_ID]

        for step in range(max_len):
            if tokens.size(1) >= self.max_positions:
                break
            pad_mask = torch.zeros_like(tokens, dtype=torch.bool)
            logits, _ = self.forward(tokens, pad_mask)
            step_logits = logits[:, -1, :] / max(temperature, 1e-6)
            step_logits[:, blocked] = float("-inf")
            if step < config.EOS_MIN_STEP:
                step_logits[:, config.EOS_ID] = float("-inf")
            step_logits = filter_logits(step_logits, top_k, top_p)
            probabilities = F.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).squeeze(1)
            next_token = torch.where(finished, torch.full_like(next_token, config.PAD_ID), next_token)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(config.EOS_ID)
            if bool(finished.all()):
                break

        return tokens.tolist()
