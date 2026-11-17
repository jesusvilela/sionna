"""Replay frozen, byte-identical LMMSE witnesses across PyTorch CPU runtimes.

The committed input fixture stores each complex tensor as interleaved raw
little-endian bytes. This script never uses random numbers or recomputes the
frame transport to construct its primary test inputs. Reports retain the
original/transported mpmath (100-digit) reference and raw solver outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from lmmse_conditioning_sweep import (
    _error,
    _posterior_oracle,
    _raw_output,
    _raw_reference,
    _tensor_diagnostics,
)
from sionna.phy.mimo.equalization import lmmse_equalizer


TENSOR_KEYS = ("y", "h", "s", "yp", "hp", "sp", "w")
DTYPE_MAP = {"complex64": np.dtype("<c8"), "complex128": np.dtype("<c16")}


def _load_tensor(entry):
    dtype = entry["dtype"]
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype {dtype!r}")
    content = bytes.fromhex(entry["raw_le_hex"])
    sha = hashlib.sha256(content).hexdigest()
    if sha != entry["sha256"]:
        raise AssertionError("Frozen fixture tensor hash mismatch")
    shape = entry["shape"]
    actual = np.frombuffer(content, dtype=DTYPE_MAP[dtype]).copy()
    if actual.size != int(np.prod(shape)):
        raise AssertionError("Incorrect frozen tensor element count")
    tensor = torch.from_numpy(actual.reshape(shape))
    if tensor.detach().cpu().contiguous().numpy().tobytes() != content:
        raise AssertionError("Tensor deserialization did not preserve bits")
    return tensor


def _result_sha(x, v):
    return hashlib.sha256(
        x.detach().cpu().contiguous().numpy().tobytes()
        + v.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _replay_one(case):
    params = case["parameters"]
    tensors = {key: _load_tensor(case["tensors"][key]) for key in TENSOR_KEYS}
    original = (tensors["y"], tensors["h"], tensors["s"])
    transported = (tensors["yp"], tensors["hp"], tensors["sp"])
    w = tensors["w"]
    tol = 2e-4 if params["precision"] == "single" else 2e-11
    safe = params["kappa_s"] <= 30 and params["kappa_w"] <= 3
    base_diagnostic = _tensor_diagnostics(original[2], w)
    transported_diagnostic = _tensor_diagnostics(transported[2], w)
    transformed_again = w @ original[2] @ w.mH
    residual = (transformed_again - transported[2]).abs().max().item()
    reconstruction_sha = hashlib.sha256(
        transformed_again.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()

    ref_base = _raw_reference(*_posterior_oracle(*original))
    ref_trans = _raw_reference(*_posterior_oracle(*transported))
    oracle_drift = _error(ref_trans, ref_base, tol)

    out = []
    for whitening in (False, True):
        row = {
            "id": case["id"],
            "parameters": params,
            "whiten_interference": whitening,
            "safe_gate": safe,
            "base_diagnostics": base_diagnostic,
            "transported_diagnostics": transported_diagnostic,
            "recomputed_transport_max_abs": residual,
            "recomputed_transport_bytes_equal": (
                reconstruction_sha == case["tensors"]["sp"]["sha256"]
            ),
            "recomputed_transport_sha256": reconstruction_sha,
            "original_oracle": ref_base,
            "transported_oracle": ref_trans,
            "input_roundoff": oracle_drift,
        }
        try:
            xb, vb = lmmse_equalizer(
                *original, whiten_interference=whitening,
                precision=params["precision"]
            )
            xt, vt = lmmse_equalizer(
                *transported, whiten_interference=whitening,
                precision=params["precision"]
            )
            if not (torch.isfinite(xb).all() and torch.isfinite(vb).all()
                    and torch.isfinite(xt).all() and torch.isfinite(vt).all()):
                raise FloatingPointError("nonfinite result")
            base_result = _raw_output(xb, vb)
            transformed_result = _raw_output(xt, vt)
            row.update({
                "state": "finite",
                "original_output": base_result,
                "transported_output": transformed_result,
                "original_result_sha256": _result_sha(xb, vb),
                "transformed_result_sha256": _result_sha(xt, vt),
                "original_solver_error": _error(base_result, ref_base, tol),
                "transported_solver_error": _error(
                    transformed_result, ref_trans, tol
                ),
                "end_to_end_frame_error": _error(
                    transformed_result, base_result, tol
                ),
            })
        except Exception as exc:
            row["state"] = "solver_error"
            row["exception"] = repr(exc)
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    opts = parser.parse_args()
    if sys.byteorder != "little":
        raise RuntimeError("This fixture format requires little-endian host")
    if opts.repeats < 2:
        raise ValueError("Use multiple replay attempts to observe repeatability")

    torch.set_num_threads(1)
    fixture_bytes = Path(opts.fixtures).read_bytes()
    fixtures = json.loads(fixture_bytes)
    if fixtures["header"]["schema"] != "lmmse-replay-v1":
        raise ValueError("Unrecognized fixture schema")
    if len(fixtures["cases"]) != fixtures["header"]["selected_cases"]:
        raise AssertionError("Fixture count disagreement")

    rows = []
    for repeat in range(opts.repeats):
        for case in fixtures["cases"]:
            for row in _replay_one(case):
                row["repeat"] = repeat
                rows.append(row)

    stable = {}
    for case in fixtures["cases"]:
        key = case["id"]
        hashes = {}
        for whitening in (False, True):
            subset = [
                r for r in rows
                if r["id"] == key and r["whiten_interference"] == whitening
            ]
            tokens = [
                (r.get("state"), r.get("original_result_sha256"),
                 r.get("transformed_result_sha256"))
                for r in subset
            ]
            hashes[str(whitening)] = {
                "within_runtime_bitwise_stable": len(set(tokens)) == 1,
                "tokens": tokens,
            }
        stable[key] = hashes

    good = [r for r in rows if r["state"] == "finite"]
    safe = [r for r in rows if r["safe_gate"]]
    safe_bad = [
        r for r in safe if r["state"] != "finite"
        or r["original_solver_error"]["max_tolerance_ratio"] > 1
        or r["transported_solver_error"]["max_tolerance_ratio"] > 1
        or r["end_to_end_frame_error"]["max_tolerance_ratio"] > 1
    ]
    summary = {
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "case_ids": [x["id"] for x in fixtures["cases"]],
        "tested_rows": len(rows),
        "safe_rows": len(safe),
        "safe_failures": len(safe_bad),
        "nonfinite_rows": len(rows) - len(good),
        "extreme_finite_out_of_tolerance": sum(
            r["state"] == "finite" and not r["safe_gate"] and (
                r["original_solver_error"]["max_tolerance_ratio"] > 1
                or r["transported_solver_error"]["max_tolerance_ratio"] > 1
                or r["end_to_end_frame_error"]["max_tolerance_ratio"] > 1
            )
            for r in rows
        ),
        "within_runtime_bitwise_results": stable,
        "fixture_input_sha256": {
            c["id"]: {k: c["tensors"][k]["sha256"] for k in TENSOR_KEYS}
            for c in fixtures["cases"]
        },
        "cholesky_status": {
            r["id"]: {
                "original": r["base_diagnostics"]["cholesky_info"],
                "transported": r["transported_diagnostics"]["cholesky_info"],
            }
            for r in rows
        },
    }
    report = {
        "protocol": {
            "experiment": "fixed represented-input LMMSE replay",
            "source": opts.fixtures,
            "repeats": opts.repeats,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cpu": platform.processor(),
            "architecture": platform.machine(),
            "threads": torch.get_num_threads(),
            "scope": "synthetic CPU linear algebra, not radio calibration",
        },
        "summary": summary,
        "rows": rows,
    }
    path = Path(opts.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8"
    )
    public = {k: v for k, v in summary.items() if k != "fixture_input_sha256"}
    # Full rows remain in the artifact; log stays human-inspectable.
    print("FROZEN_LMMSE_REPLAY_SUMMARY=" + json.dumps(public, sort_keys=True))
    if safe_bad:
        raise SystemExit(f"Safe fixture gate failed in {len(safe_bad)} rows")


if __name__ == "__main__":
    main()
