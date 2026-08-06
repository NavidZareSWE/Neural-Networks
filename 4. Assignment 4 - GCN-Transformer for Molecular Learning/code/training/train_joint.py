import copy

import torch

import config
from training.losses import (lm_loss_and_accuracy, info_nce,
                             positive_pair_loss, perplexity)


@torch.no_grad()
def evaluate_joint(gcn, transformer, loader, device):
    gcn.eval(); transformer.eval()
    totals = {"lm": 0.0, "con": 0.0, "pair": 0.0, "tok_acc": 0.0, "ret_acc": 0.0}
    batches = 0
    for graphs, tokens, pad_mask, eos_positions in loader:
        graphs = graphs.to(device)
        tokens, pad_mask, eos_positions = tokens.to(device), pad_mask.to(device), eos_positions.to(device)
        logits, hidden = transformer(tokens, pad_mask)
        lm, tok_acc = lm_loss_and_accuracy(logits, tokens)
        graph_embed = gcn(graphs)
        sequence_embed = transformer.project_eos(hidden, eos_positions)
        con, ret_acc = info_nce(graph_embed, sequence_embed)
        pair = positive_pair_loss(graph_embed, sequence_embed)
        totals["lm"] += lm.item(); totals["con"] += con.item(); totals["pair"] += pair.item()
        totals["tok_acc"] += tok_acc; totals["ret_acc"] += ret_acc; batches += 1
    batches = max(batches, 1)
    return {k: v / batches for k, v in totals.items()}


def train_joint(gcn, transformer, train_loader, val_loader, device=None,
                epochs=None, lr=None):
    device = device or config.DEVICE
    epochs = epochs or config.JOINT_EPOCHS
    lr = lr or config.LR
    contrastive_weight = config.CONTRASTIVE_WEIGHT
    pair_weight = config.PAIR_WEIGHT

    gcn.to(device); transformer.to(device)
    parameters = list(gcn.parameters()) + list(transformer.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)

    history = {"train_total": [], "train_lm": [], "train_contrastive": [], "train_pair": [],
               "val_lm": [], "val_contrastive": [], "val_pair": [], "val_tok_acc": [],
               "val_retrieval_acc": [], "val_ppl": []}

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        gcn.train(); transformer.train()
        run_total, run_lm, run_con, run_pair, batches = 0.0, 0.0, 0.0, 0.0, 0
        for graphs, tokens, pad_mask, eos_positions in train_loader:
            graphs = graphs.to(device)
            tokens, pad_mask, eos_positions = tokens.to(device), pad_mask.to(device), eos_positions.to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda", enabled=config.USE_AMP):
                logits, hidden = transformer(tokens, pad_mask)
                lm, _ = lm_loss_and_accuracy(logits, tokens)
                graph_embed = gcn(graphs)
                sequence_embed = transformer.project_eos(hidden, eos_positions)
                con, _ = info_nce(graph_embed, sequence_embed)
                pair = positive_pair_loss(graph_embed, sequence_embed)
                loss = lm + contrastive_weight * con + pair_weight * pair
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            run_total += loss.item(); run_lm += lm.item()
            run_con += con.item(); run_pair += pair.item(); batches += 1
        scheduler.step()

        batches = max(batches, 1)
        val = evaluate_joint(gcn, transformer, val_loader, device)
        val_total = val["lm"] + contrastive_weight * val["con"] + pair_weight * val["pair"]
        val_ppl = perplexity(val["lm"])

        history["train_total"].append(float(run_total / batches))
        history["train_lm"].append(float(run_lm / batches))
        history["train_contrastive"].append(float(run_con / batches))
        history["train_pair"].append(float(run_pair / batches))
        history["val_lm"].append(float(val["lm"]))
        history["val_contrastive"].append(float(val["con"]))
        history["val_pair"].append(float(val["pair"]))
        history["val_tok_acc"].append(float(val["tok_acc"]))
        history["val_retrieval_acc"].append(float(val["ret_acc"]))
        history["val_ppl"].append(float(val_ppl))

        print(f"[joint {epoch:02d}/{epochs}] total={run_total / batches:.4f} "
              f"lm={run_lm / batches:.4f} con={run_con / batches:.4f} pair={run_pair / batches:.4f} "
              f"| val_total={val_total:.4f} val_tok_acc={val['tok_acc']:.4f} "
              f"val_ret_acc={val['ret_acc']:.4f} val_ppl={val_ppl:.3f}")

        if val_total < best_val - 1e-4:
            best_val = val_total
            best_state = (copy.deepcopy(gcn.state_dict()), copy.deepcopy(transformer.state_dict()))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{config.EARLY_STOP_PATIENCE} epochs).")
                break

    if best_state is not None:
        gcn.load_state_dict(best_state[0])
        transformer.load_state_dict(best_state[1])

    return history
