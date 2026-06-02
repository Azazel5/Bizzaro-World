from __future__ import annotations

from functools import partial
from itertools import product
from typing import Any, Callable

import torch as t
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer, utils

from .hooks import patch_head_input, patch_or_freeze_head_vectors
from .metrics import compute_logit_diff


def _get_baseline_logit_diffs(model: HookedTransformer, dataset: Any) -> tuple[t.Tensor, t.Tensor]:
    with t.no_grad():
        clean_logits = model(dataset.clean_toks, return_type="logits")
        corrupt_logits = model(dataset.corrupt_toks, return_type="logits")
    return compute_logit_diff(clean_logits, dataset), compute_logit_diff(corrupt_logits, dataset)


def get_path_patch_head_to_final_resid_post(
    model: HookedTransformer,
    dataset: Any,
    patching_metric: Callable,
    clean_cache: ActivationCache | None = None,
    corrupt_cache: ActivationCache | None = None,
) -> t.Tensor:
    """
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = final value of residual stream

    Returns:
        tensor of metric values for every possible sender head
    """
    model.reset_hooks()
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=model.cfg.device, dtype=t.float32)

    z_name_filter = lambda name: name.endswith("z")
    resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_post_name_filter = lambda name: name == resid_post_hook_name

    if clean_cache is None:
        _, clean_cache = model.run_with_cache(dataset.clean_toks, names_filter=z_name_filter, return_type=None)
    if corrupt_cache is None:
        _, corrupt_cache = model.run_with_cache(dataset.corrupt_toks, names_filter=z_name_filter, return_type=None)

    clean_ld, corrupt_ld = _get_baseline_logit_diffs(model, dataset)
    metric_fn = partial(patching_metric, dataset=dataset, clean_ld=clean_ld, corrupt_ld=corrupt_ld)

    for sender_layer, sender_head in tqdm(list(product(range(model.cfg.n_layers), range(model.cfg.n_heads)))):
        hook_fn = partial(
            patch_or_freeze_head_vectors,
            new_cache=corrupt_cache,
            orig_cache=clean_cache,
            head_to_patch=(sender_layer, sender_head),
        )
        model.add_hook(z_name_filter, hook_fn, level=1)

        _, patched_cache = model.run_with_cache(
            dataset.clean_toks, names_filter=resid_post_name_filter, return_type=None
        )
        assert set(patched_cache.keys()) == {resid_post_hook_name}

        patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook_name]))
        results[sender_layer, sender_head] = metric_fn(patched_logits)

        model.reset_hooks()

    return results


def get_path_patch_head_to_heads(
    receiver_heads: list[tuple[int, int]],
    receiver_input: str,
    model: HookedTransformer,
    dataset: Any,
    patching_metric: Callable,
    clean_cache: ActivationCache | None = None,
    corrupt_cache: ActivationCache | None = None,
) -> t.Tensor:
    """
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = input to a later head (or set of heads)

    The receiver node is specified by receiver_heads and receiver_input.

    Returns:
        tensor of metric values for every possible sender head
    """
    model.reset_hooks()

    assert receiver_input in ("k", "q", "v")
    receiver_layers = set(next(zip(*receiver_heads)))
    receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    results = t.zeros(max(receiver_layers), model.cfg.n_heads, device=model.cfg.device, dtype=t.float32)
    z_name_filter = lambda name: name.endswith("z")

    if clean_cache is None:
        _, clean_cache = model.run_with_cache(dataset.clean_toks, names_filter=z_name_filter, return_type=None)
    if corrupt_cache is None:
        _, corrupt_cache = model.run_with_cache(dataset.corrupt_toks, names_filter=z_name_filter, return_type=None)

    clean_ld, corrupt_ld = _get_baseline_logit_diffs(model, dataset)
    metric_fn = partial(patching_metric, dataset=dataset, clean_ld=clean_ld, corrupt_ld=corrupt_ld)

    for sender_layer, sender_head in tqdm(list(product(range(max(receiver_layers)), range(model.cfg.n_heads)))):
        hook_fn = partial(
            patch_or_freeze_head_vectors,
            new_cache=corrupt_cache,
            orig_cache=clean_cache,
            head_to_patch=(sender_layer, sender_head),
        )
        model.add_hook(z_name_filter, hook_fn, level=1)

        _, patched_cache = model.run_with_cache(
            dataset.clean_toks, names_filter=receiver_hook_names_filter, return_type=None
        )
        assert set(patched_cache.keys()) == set(receiver_hook_names)

        hook_fn = partial(
            patch_head_input,
            patched_cache=patched_cache,
            head_list=receiver_heads,
        )
        patched_logits = model.run_with_hooks(
            dataset.clean_toks,
            fwd_hooks=[(receiver_hook_names_filter, hook_fn)],
            return_type="logits",
        )

        results[sender_layer, sender_head] = metric_fn(patched_logits)
        model.reset_hooks()

    return results