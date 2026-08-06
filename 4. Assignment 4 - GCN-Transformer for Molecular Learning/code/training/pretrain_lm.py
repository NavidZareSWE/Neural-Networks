import torch

import config
from training.losses import lm_loss_and_accuracy, perplexity


@torch.no_grad()
def evaluate_lm(transformer, loader, device):
    transformer.eval()
    total_loss, total_acc, batches = 0.0, 0.0, 0
    for _, tokens, pad_mask, _ in loader:
        tokens, pad_mask = tokens.to(device), pad_mask.to(device)
        logits, _ = transformer(tokens, pad_mask)
        loss, acc = lm_loss_and_accuracy(logits, tokens)
        total_loss += loss.item(); total_acc += acc; batches += 1
    return total_loss / max(batches, 1), total_acc / max(batches, 1)


def pretrain_language_model(transformer, train_loader, val_loader, device=None,
                            epochs=None, lr=None):
    device = device or config.DEVICE
    epochs = epochs or config.LM_EPOCHS
    lr = lr or config.LR

    transformer.to(device)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr,
                                  weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)

    history = {"train_loss": [], "val_loss": [], "train_acc": [],
               "val_acc": [], "val_ppl": []}

    for epoch in range(1, epochs + 1):
        transformer.train()
        running_loss, running_acc, batches = 0.0, 0.0, 0
        for _, tokens, pad_mask, _ in train_loader:
            tokens, pad_mask = tokens.to(device), pad_mask.to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda", enabled=config.USE_AMP):
                logits, _ = transformer(tokens, pad_mask)
                loss, acc = lm_loss_and_accuracy(logits, tokens)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item(); running_acc += acc; batches += 1

        train_loss = running_loss / max(batches, 1)
        train_acc = running_acc / max(batches, 1)
        val_loss, val_acc = evaluate_lm(transformer, val_loader, device)
        val_ppl = perplexity(val_loss)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_ppl"].append(float(val_ppl))

        print(f"[LM warm-up {epoch:02d}/{epochs}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_tok_acc={val_acc:.4f} val_ppl={val_ppl:.3f}")

    return history
