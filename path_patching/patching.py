from __future__ import annotations

from functools import partial
from itertools import product
from typing import Any, Callable

import torch as t
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer, utils

from hooks import patch_head_input, patch_or_freeze_head_vectors


def get_path_patch_head_to_final_resid_post(
    model: HookedTransformer,
    dataset: Any,
    patching_metric: Callable,
    clean_cache: ActivationCache | None = None,
    corrupt_cache: ActivationCache | None = None,
    checkpoint_path: str | Path | None = None,
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
        results[sender_layer, sender_head] = patching_metric(patched_logits)

        if checkpoint_path is not None:
            t.save(results.detach().cpu(), checkpoint_path)

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
    checkpoint_path: str | Path | None = None,
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
    if not receiver_heads:
        return t.zeros(0, model.cfg.n_heads, device=model.cfg.device, dtype=t.float32)

    receiver_layers = set(next(zip(*receiver_heads)))
    receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    results = t.zeros(max(receiver_layers), model.cfg.n_heads, device=model.cfg.device, dtype=t.float32)
    z_name_filter = lambda name: name.endswith("z")

    if clean_cache is None:
        _, clean_cache = model.run_with_cache(dataset.clean_toks, names_filter=z_name_filter, return_type=None)
    if corrupt_cache is None:
        _, corrupt_cache = model.run_with_cache(dataset.corrupt_toks, names_filter=z_name_filter, return_type=None)

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

        results[sender_layer, sender_head] = patching_metric(patched_logits)

        if checkpoint_path is not None:
            t.save(results.detach().cpu(), checkpoint_path)

        model.reset_hooks()

    return results

def get_path_patch_sender_to_heads(
    sender_head: tuple[int, int],
    receiver_input: str,
    model: HookedTransformer,
    dataset: Any,
    patching_metric: Callable,
    clean_cache: ActivationCache | None = None,
    corrupt_cache: ActivationCache | None = None,
    checkpoint_path=None,
) -> t.Tensor:
    """
    Path patching with a fixed sender head, sweeping all downstream heads as receivers.

    Mirror of get_path_patch_head_to_heads: fixes the sender, sweeps every
    (layer, head) pair in layers strictly after the sender as receivers.

    Returns:
        Tensor of shape [n_layers - sender_layer - 1, n_heads].
        Row k corresponds to layer (sender_layer + 1 + k).
    """
    sender_layer, sender_head_idx = sender_head
    n_downstream = model.cfg.n_layers - sender_layer - 1

    if n_downstream <= 0:
        return t.zeros(0, model.cfg.n_heads, device=model.cfg.device, dtype=t.float32)

    assert receiver_input in ("k", "q", "v")
    results = t.zeros(n_downstream, model.cfg.n_heads, device=model.cfg.device, dtype=t.float32)
    z_name_filter = lambda name: name.endswith("z")

    if clean_cache is None:
        _, clean_cache = model.run_with_cache(dataset.clean_toks, names_filter=z_name_filter, return_type=None)
    if corrupt_cache is None:
        _, corrupt_cache = model.run_with_cache(dataset.corrupt_toks, names_filter=z_name_filter, return_type=None)

    for receiver_layer in tqdm(range(sender_layer + 1, model.cfg.n_layers)):
        receiver_hook_name = utils.get_act_name(receiver_input, receiver_layer)
        # Default-arg capture avoids late-binding closure across loop iterations
        is_recv_hook = lambda name, _h=receiver_hook_name: name == _h

        # Step 1: patch sender z-output (freeze all others), cache this receiver layer's activations
        model.add_hook(
            z_name_filter,
            partial(
                patch_or_freeze_head_vectors,
                new_cache=corrupt_cache,
                orig_cache=clean_cache,
                head_to_patch=(sender_layer, sender_head_idx),
            ),
            level=1,
        )
        _, patched_cache = model.run_with_cache(
            dataset.clean_toks, names_filter=is_recv_hook, return_type=None
        )
        model.reset_hooks()

        # Step 2: patch each head's receiver input individually and measure
        row_idx = receiver_layer - sender_layer - 1
        for recv_head_idx in range(model.cfg.n_heads):
            patched_logits = model.run_with_hooks(
                dataset.clean_toks,
                fwd_hooks=[
                    (
                        is_recv_hook,
                        partial(
                            patch_head_input,
                            patched_cache=patched_cache,
                            head_list=[(receiver_layer, recv_head_idx)],
                        ),
                    )
                ],
                return_type="logits",
            )
            results[row_idx, recv_head_idx] = patching_metric(patched_logits)

        if checkpoint_path is not None:
            t.save(results.detach().cpu(), checkpoint_path)

    model.reset_hooks()
    return results
