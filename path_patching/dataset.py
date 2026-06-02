from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    import torch
    from torch.utils.data import Dataset

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for environments without torch
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[override]
        pass

    _TORCH_AVAILABLE = False

from shared.fact_battery import load_fact_battery


class _SimpleTensor:
    def __init__(self, data: Any, shape: Tuple[int, ...] | None = None):
        self._data = data
        self._shape = shape

    @property
    def shape(self) -> Tuple[int, ...]:
        if self._shape is not None:
            return self._shape
        return _infer_shape(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        for item in self._data:
            yield item

    def __getitem__(self, idx):
        value = self._data[idx]
        if isinstance(value, list):
            return _SimpleTensor(value, _infer_shape(value))
        return value

    def tolist(self) -> Any:
        return self._data

    def __repr__(self) -> str:
        return f"_SimpleTensor(shape={self.shape}, data={self._data!r})"


def _infer_shape(data: Any) -> Tuple[int, ...]:
    if not isinstance(data, list):
        return ()
    if not data:
        return (0,)
    if isinstance(data[0], list):
        inner = _infer_shape(data[0])
        return (len(data),) + inner
    return (len(data),)


def _make_1d_tensor(values: Sequence[int]):
    if _TORCH_AVAILABLE:
        return torch.tensor(list(values), dtype=torch.long)
    return _SimpleTensor(list(values), (len(values),))


def _make_2d_tensor(sequences: Sequence[Sequence[int]]):
    if _TORCH_AVAILABLE:
        if not sequences:
            return torch.empty((0, 0), dtype=torch.long)
        return torch.tensor([list(seq) for seq in sequences], dtype=torch.long)

    rows = [list(seq) for seq in sequences]
    width = max((len(row) for row in rows), default=0)
    return _SimpleTensor(rows, (len(rows), width))


def _encode(tokenizer: Any, text: str) -> List[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _token_piece(tokenizer: Any, token_id: int) -> str:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        pieces = tokenizer.convert_ids_to_tokens([token_id])
        if pieces:
            return str(pieces[0])
    return tokenizer.decode([token_id], skip_special_tokens=True)


def _normalize_piece(piece: str) -> str:
    return (
        piece.replace("Ġ", "")
        .replace("▁", "")
        .replace("##", "")
        .strip()
        .lower()
    )


def _single_token_id(tokenizer: Any, text: str) -> int:
    token_ids = _encode(tokenizer, text)
    if len(token_ids) != 1:
        raise ValueError(f"Expected a single token for {text!r}, got {token_ids}")
    return token_ids[0]


def _pad_id(tokenizer: Any) -> int:
    for attr in ("pad_token_id", "eos_token_id", "unk_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return 0


def _pad_encoded_sequences(sequences: Sequence[Sequence[int]], pad_id: int):
    if not sequences:
        if _TORCH_AVAILABLE:
            return torch.empty((0, 0), dtype=torch.long)
        return _SimpleTensor([], (0, 0))

    max_len = max(len(seq) for seq in sequences)
    padded = [list(seq) + [pad_id] * (max_len - len(seq)) for seq in sequences]
    return _make_2d_tensor(padded)


def _find_entity_position(tokenizer: Any, prompt: str, entity_token: str) -> int:
    target = entity_token.strip().lower()
    token_ids = _encode(tokenizer, prompt)

    normalized_pieces = [_normalize_piece(_token_piece(tokenizer, token_id)) for token_id in token_ids]
    for idx, piece in enumerate(normalized_pieces):
        if target in piece:
            return idx

    for start in range(len(normalized_pieces)):
        joined = "".join(normalized_pieces[start:])
        if target and target in joined:
            return start

    raise ValueError(
        f"Could not locate entity token {entity_token!r} in prompt {prompt!r}"
    )


def check_single_token_targets(json_path: str | Path, tokenizer: Any) -> List[Dict[str, str]]:
    path = Path(json_path)
    prompts = load_fact_battery(path)

    multi_clean = 0
    multi_corrupt = 0
    multi_any = 0
    filtered: List[Dict[str, str]] = []

    for entry in prompts:
        clean_ids = _encode(tokenizer, entry["clean_target"])
        corrupt_ids = _encode(tokenizer, entry["corrupt_target"])
        clean_ok = len(clean_ids) == 1
        corrupt_ok = len(corrupt_ids) == 1

        if not clean_ok:
            multi_clean += 1
        if not corrupt_ok:
            multi_corrupt += 1
        if not (clean_ok and corrupt_ok):
            multi_any += 1
            continue

        filtered.append(entry)

    print(f"multi-token clean targets: {multi_clean}")
    print(f"multi-token corrupt targets: {multi_corrupt}")
    print(f"entries with any multi-token target: {multi_any}")
    print(f"kept single-token entries: {len(filtered)}/{len(prompts)}")
    return filtered


class FactualRecallDataset(Dataset):
    def __init__(self, json_path: str | Path, tokenizer: Any):
        self.json_path = Path(json_path)
        self.tokenizer = tokenizer
        self.prompts = load_fact_battery(self.json_path)

        pad_id = _pad_id(self.tokenizer)
        self.clean_toks = _pad_encoded_sequences(
            [_encode(self.tokenizer, prompt["clean_prompt"]) for prompt in self.prompts],
            pad_id,
        )
        self.corrupt_toks = _pad_encoded_sequences(
            [_encode(self.tokenizer, prompt["corrupt_prompt"]) for prompt in self.prompts],
            pad_id,
        )

        io_ids = [_single_token_id(self.tokenizer, prompt["clean_target"]) for prompt in self.prompts]
        s_ids = [_single_token_id(self.tokenizer, prompt["corrupt_target"]) for prompt in self.prompts]
        self.io_tokenIDs = _make_1d_tensor(io_ids)
        self.s_tokenIDs = _make_1d_tensor(s_ids)

        entity_positions = [
            _find_entity_position(self.tokenizer, prompt["clean_prompt"], prompt["entity_token"])
            for prompt in self.prompts
        ]
        self.word_idx = {
            "entity": _make_1d_tensor(entity_positions),
            "end": _make_1d_tensor([-1] * len(self.prompts)),
        }

        self.groups: List[List[int]] = self._group_indices_by_category()

    def _group_indices_by_category(self) -> List[List[int]]:
        groups_by_category: Dict[str, List[int]] = {}
        for idx, prompt in enumerate(self.prompts):
            groups_by_category.setdefault(prompt["category"], []).append(idx)
        return list(groups_by_category.values())

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "clean_toks": self.clean_toks[idx],
            "corrupt_toks": self.corrupt_toks[idx],
            "io_tokenID": self.io_tokenIDs[idx],
            "s_tokenID": self.s_tokenIDs[idx],
            "word_idx": {
                "entity": self.word_idx["entity"][idx],
                "end": self.word_idx["end"][idx],
            },
            "prompt": self.prompts[idx],
        }