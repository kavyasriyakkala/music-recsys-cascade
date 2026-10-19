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
    # Use log(freq) rather than log(freq/sum): the constant log(sum) adds the
    # same large value to every off-diagonal logit, inflating the loss by ~log(N)
    # without changing gradients.  Masking the diagonal prevents corrupting the
    # positive-pair logit.
    log_freqs = torch.log(item_freqs.clamp(min=1.0))       # [B]
    correction = log_freqs.unsqueeze(0).expand_as(logits)  # [B, B]
    diag_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
    logits = logits - correction.masked_fill(diag_mask, 0.0)

    targets = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, targets)
