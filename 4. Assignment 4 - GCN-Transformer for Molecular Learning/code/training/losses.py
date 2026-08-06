import math

import torch
import torch.nn.functional as F

import config


def info_nce(graph_embed, sequence_embed, tau=None):
    tau = config.CONTRASTIVE_TAU if tau is None else tau
    logits = graph_embed @ sequence_embed.t() / tau
    labels = torch.arange(graph_embed.size(0), device=graph_embed.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    with torch.no_grad():
        acc = 0.5 * ((logits.argmax(1) == labels).float().mean().item()
                     + (logits.t().argmax(1) == labels).float().mean().item())
    return loss, acc


def positive_pair_loss(graph_embed, sequence_embed):
    cosine = (graph_embed * sequence_embed).sum(dim=-1)
    return (1.0 - cosine).mean()


def lm_loss_and_accuracy(logits, tokens, pad_id=None):
    pad_id = config.PAD_ID if pad_id is None else pad_id
    prediction = logits[:, :-1, :].contiguous()
    target = tokens[:, 1:].contiguous()
    loss = F.cross_entropy(prediction.view(-1, prediction.size(-1)),
                           target.view(-1), ignore_index=pad_id)
    with torch.no_grad():
        valid = target != pad_id
        correct = (prediction.argmax(dim=-1) == target) & valid
        accuracy = correct.sum().item() / max(valid.sum().item(), 1)
    return loss, accuracy


def perplexity(loss_value):
    return math.exp(min(loss_value, 20.0))
