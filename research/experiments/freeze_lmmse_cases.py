"""Freeze actual PyTorch case tensors for byte-identical cross-runtime replay.

A one-time producer imports the historical adversarial-case generator.
The printed FROZEN_LMMSE_CASE lines are committed verbatim as a JSON fixture,
so subsequent replay never depends on random draws, QR, or Torch versions.
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from lmmse_conditioning_sweep import _case

CASES = (
    ("safe_float32_2x2", "single", 2, 30.0, 3.0, 0),
    ("failure_float32_2x2_w30", "single", 2, 1e6, 30.0, 1),
    ("failure_float32_2x2_w300", "single", 2, 1e6, 300.0, 0),
    ("failure_float32_4x4_w300", "single", 4, 1e6, 300.0, 1),
    ("safe_float64_2x2", "double", 2, 30.0, 3.0, 0),
    ("extreme_float64_2x2_w300", "double", 2, 1e6, 300.0, 0),
)


def freeze_tensor(tensor):
    raw = np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes(order="C")
    return {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(tensor.shape),
        "raw_le_hex": raw.hex(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main():
    if sys.byteorder != "little":
        raise RuntimeError("Fixture encoding explicitly requires little-endian CPU")
    torch.set_num_threads(1)
    header = {
        "schema": "lmmse-replay-v1",
        "provenance": "one-time PyTorch QR/randn producer, NOT independent RF data",
        "generator": "research/experiments/lmmse_conditioning_sweep.py::_case",
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.machine(),
        "dtype_storage": "complex NumPy C-contiguous interleaved little-endian",
        "selected_cases": len(CASES),
    }
    print("FROZEN_LMMSE_HEADER=" + json.dumps(header, sort_keys=True))
    output = {"header": header, "cases": []}
    for label, precision, dim, kappa_s, kappa_w, seed in CASES:
        original, transformed, w = _case(
            precision, dim, kappa_s, kappa_w, seed
        )
        item = {
            "id": label,
            "parameters": {
                "precision": precision,
                "dim": dim,
                "kappa_s": kappa_s,
                "kappa_w": kappa_w,
                "seed": seed,
            },
            "tensors": {
                "y": freeze_tensor(original[0]),
                "h": freeze_tensor(original[1]),
                "s": freeze_tensor(original[2]),
                "yp": freeze_tensor(transformed[0]),
                "hp": freeze_tensor(transformed[1]),
                "sp": freeze_tensor(transformed[2]),
                "w": freeze_tensor(w),
            },
        }
        output["cases"].append(item)
        print("FROZEN_LMMSE_CASE=" + json.dumps(item, sort_keys=True))
    out = Path("research/results/frozen_input_producer.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("FROZEN_LMMSE_CASE_COUNT=" + str(len(output["cases"])))


if __name__ == "__main__":
    main()
