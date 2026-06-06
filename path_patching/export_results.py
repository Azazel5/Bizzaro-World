from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _serializable(obj: Any):
    # Convert basic non-serializable types to serializable forms
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.detach().cpu().item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, (float, int, str, bool)):
        return obj
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return str(obj)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _tensor_summary(t: torch.Tensor) -> dict:
    t_cpu = t.detach().cpu()
    numel = int(t_cpu.numel())
    summary = {
        "shape": list(t_cpu.shape),
        "numel": numel,
        "dtype": str(t_cpu.dtype),
    }
    if numel > 0:
        try:
            summary.update(
                {
                    "mean": float(t_cpu.float().mean().item()),
                    "std": float(t_cpu.float().std().item()),
                    "min": float(t_cpu.min().item()),
                    "max": float(t_cpu.max().item()),
                }
            )
        except Exception:
            # Fallback: convert to list and compute in Python
            vals = t_cpu.view(-1).tolist()
            summary.update({
                "mean": sum(vals) / len(vals),
                "std": 0.0,
                "min": min(vals),
                "max": max(vals),
            })
    return summary


def export_results(results_dir: Path | str, max_elements: int = 100_000) -> None:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    # Baseline metrics
    bm_path = results_dir / "baseline_metrics.pt"
    if bm_path.exists():
        bm = torch.load(bm_path)
        out = {}
        for k, v in bm.items():
            out[k] = _serializable(v)
        _write_json(results_dir / "baseline_metrics.json", out)

    # Tensor artifacts to summarize
    tensor_files = [
        ("path_patch_final_resid.pt", "path_patch_final_resid_summary.json", "path_patch_final_resid.json"),
        ("path_patch_heads_q.pt", "path_patch_heads_q_summary.json", "path_patch_heads_q.json"),
    ]

    for fname, summary_name, full_name in tensor_files:
        p = results_dir / fname
        if not p.exists():
            continue
        t = torch.load(p)
        # allow the saved payload to be non-tensor (e.g., dict of results)
        if isinstance(t, torch.Tensor):
            summary = _tensor_summary(t)
            _write_json(results_dir / summary_name, summary)
            if summary["numel"] <= max_elements:
                # safe to export full array
                _write_json(results_dir / full_name, {"shape": summary["shape"], "values": t.detach().cpu().tolist()})
        else:
            # try to make it serializable
            try:
                serial = _serializable(t)
                _write_json(results_dir / summary_name, {"type": type(t).__name__, "value": serial})
            except Exception:
                _write_json(results_dir / summary_name, {"type": type(t).__name__, "value": str(t)})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export .pt artifacts to JSON summaries")
    parser.add_argument("--results-dir", required=True, help="Directory containing result .pt files")
    parser.add_argument(
        "--max-elements",
        type=int,
        default=100_000,
        help="Maximum number of elements to allow full-array JSON export",
    )
    args = parser.parse_args()
    export_results(args.results_dir, max_elements=args.max_elements)
