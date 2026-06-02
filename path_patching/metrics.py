from __future__ import annotations

from typing import Any

import torch as t


def compute_logit_diff(logits: t.Tensor, dataset: Any) -> t.Tensor:
    final_logits = logits[:, -1]
    batch_indices = t.arange(final_logits.shape[0], device=final_logits.device)
    io_logits = final_logits[batch_indices, dataset.io_tokenIDs.to(final_logits.device)]
    s_logits = final_logits[batch_indices, dataset.s_tokenIDs.to(final_logits.device)]
    return (io_logits - s_logits).mean()


def factual_recall_metric(
    logits: t.Tensor,
    dataset: Any,
    clean_ld: t.Tensor,
    corrupt_ld: t.Tensor,
) -> t.Tensor:
    patched_ld = compute_logit_diff(logits, dataset)
    return (patched_ld - clean_ld) / (clean_ld - corrupt_ld)