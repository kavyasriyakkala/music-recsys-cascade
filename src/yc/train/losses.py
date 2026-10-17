"""Training losses for the two-tower retriever."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def in_batch_softmax_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    item_freqs: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """In-batch sampled softmax loss with log-frequency correction.

    Implements Sampling-Bias-Corrected Neural Modeling (Yi et al., 2019).
    Subtracts log(item_freq / sum(item_freqs)) from off-diagonal logits only,
    so popular items are penalized less as false negatives in random batches.

    Args:
        user_emb: [B, D] L2-normalized user embeddings.
        item_emb: [B, D] L2-normalized item embeddings.
        item_freqs: [B] frequency of each item in the training corpus.
        temperature: softmax temperature.

    Returns:
        Scalar mean cross-entropy loss.
    """
    # [B, B] similarity matrix; diagonal = (user_i, item_i) positive pairs.
    logits = user_emb @ item_emb.T / temperature

    # Log-frequency correction for off-diagonal negatives only.
    log_corr = torch.log(item_freqs / item_freqs.sum())  # [B]
    off_diag = 1.0 - torch.eye(logits.size(0), device=logits.device)
    logits = logits - log_corr.unsqueeze(0) * off_diag

    targets = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, targets)
