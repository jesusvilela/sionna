# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Bounded, deterministic conditioning audit for Sionna PHY LMMSE.

This research script reports raw solver error and input-transport rounding
separately, using an independent arbitrary-precision posterior oracle on the
ACTUAL floating-point matrices passed to each implementation. It is not a
production contribution, BER/SER experiment, or performance benchmark.

Prerecorded safe gate (before inspecting results):
  target cond(S) in {1, 30}, target cond(W) in {1, 3};
  both LMMSE implementations must agree with the represented-input mpmath
  oracle within atol=rtol=2e-4 for single, 2e-11 for double.
Outside this gate, all failures and divergences remain reported, not clipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mpmath as mp
import torch

from sionna.phy.mimo.equalization import lmmse_equalizer


COV_CONDITIONS = (1.0, 30.0, 1e3, 1e6)
FRAME_CONDITIONS = (1.0, 3.0, 30.0, 300.0)
SEEDS = (0, 1)
NUM_RX = (2, 4)
PRECISIONS = ("single", "double")
MODES = (False, True)


def _unitary(dim: int, gen: torch.Generator) -> torch.Tensor:
    z = torch.randn(dim, dim, generator=gen, dtype=torch.complex128)
    return torch.linalg.qr(z).Q


def _case(precision: str, dim: int, kappa_s: float, kappa_w: float, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(
        20261117 + 101 * seed + 17 * dim
        + 3 * COV_CONDITIONS.index(kappa_s)
        + 31 * FRAME_CONDITIONS.index(kappa_w)
    )
    cdtype = torch.complex64 if precision == "single" else torch.complex128

    u = _unitary(dim, gen)
    spec = torch.logspace(0, -torch.log10(torch.tensor(kappa_s)).item(), dim,
                          dtype=torch.float64)
    s128 = u @ torch.diag(spec.to(torch.complex128)) @ u.mH

    left = _unitary(dim, gen)
    right = _unitary(dim, gen)
    singular_values = torch.logspace(
        0, -torch.log10(torch.tensor(kappa_w)).item(), dim,
        dtype=torch.float64
    )
    w128 = left @ torch.diag(singular_values.to(torch.complex128)) @ right.mH

    # Quantize before transport, just as an actual float32/float64 tensor path.
    s = s128.to(cdtype)
    s = 0.5 * (s + s.mH)  # explicitly record this input construction
    w = w128.to(cdtype)
    h = (0.3 * torch.randn(dim, 2, generator=gen,
                             dtype=torch.complex128)).to(cdtype)
    x = torch.tensor([0.8 + 0.5j, -0.4 + 0.6j], dtype=cdtype)
    noise = (0.05 * torch.randn(dim, generator=gen,
                                dtype=torch.complex128)).to(cdtype)
    y = h @ x + noise

    yp = w @ y
    hp = w @ h
    sp = w @ s @ w.mH  # deliberately DO NOT symmetrize the transported matrix
    return (y, h, s), (yp, hp, sp), w


def _mp_matrix(tensor: torch.Tensor) -> mp.matrix:
    values = tensor.detach().cpu().tolist()
    return mp.matrix([[_mpc(v) for v in row] for row in values])


def _mpc(z) -> mp.mpc:
    return mp.mpc(mp.mpf(float(z.real)), mp.mpf(float(z.imag)))


def _mp_vector(tensor: torch.Tensor) -> mp.matrix:
    return mp.matrix([_mpc(z) for z in tensor.detach().cpu().tolist()])


def _posterior_oracle(y: torch.Tensor, h: torch.Tensor, s: torch.Tensor):
    """Independent 100-digit arithmetic for the represented input values."""
    with mp.workdps(100):
        sm = _mp_matrix(s)
        hm = _mp_matrix(h)
        ym = _mp_vector(y)
        sinv = sm ** -1
        gram = hm.transpose_conj() * sinv * hm
        posterior = (mp.eye(hm.cols) + gram) ** -1
        mean = posterior * hm.transpose_conj() * sinv * ym
        x_out = []
        v_out = []
        for i in range(hm.cols):
            d = 1 - posterior[i, i]
            x_out.append(complex(mean[i] / d))
            v_out.append(float(mp.re(1 / d - 1)))
    return x_out, v_out


def _raw_output(x: torch.Tensor, v: torch.Tensor):
    """Store original values, before any error normalization or aggregation."""
    xx = [complex(z) for z in x.detach().cpu().tolist()]
    vv = [float(z) for z in v.detach().cpu().tolist()]
    return [[z.real, z.imag] for z in xx], vv


def _raw_reference(x, v):
    return [[z.real, z.imag] for z in x], list(v)


def _error(produced, oracle, abs_tol: float):
    ax, av = produced
    ox, ov = oracle
    dx = [abs(complex(*a) - complex(*b)) for a, b in zip(ax, ox)]
    dv = [abs(a - b) for a, b in zip(av, ov)]
    norm_x = [
        diff / (abs_tol * (1.0 + abs(complex(*b))))
        for diff, b in zip(dx, ox)
    ]
    norm_v = [diff / (abs_tol * (1.0 + abs(b)))
              for diff, b in zip(dv, ov)]
    return {
        "max_abs_x": max(dx),
        "max_abs_no_eff": max(dv),
        "max_tolerance_ratio": max(norm_x + norm_v),
    }


def _tensor_diagnostics(s: torch.Tensor, w: torch.Tensor):
    s64 = s.to(torch.complex128)
    w64 = w.to(torch.complex128)
    hermiticity = (s64 - s64.mH).abs().max().item()
    scale = s64.abs().max().item()
    _, chol_info = torch.linalg.cholesky_ex(s)
    return {
        "actual_cov_condition": float(torch.linalg.cond(s64).item()),
        "actual_frame_condition": float(torch.linalg.cond(w64).item()),
        "hermiticity_raw_max": float(hermiticity),
        "covariance_raw_max": float(scale),
        "cholesky_info": int(chol_info.item()),
    }


def _trial(precision, dim, kappa_s, kappa_w, seed):
    base, transformed, w = _case(precision, dim, kappa_s, kappa_w, seed)
    tol = 2e-4 if precision == "single" else 2e-11
    safe = kappa_s <= 30 and kappa_w <= 3
    bdiag = _tensor_diagnostics(base[2], w)
    tdiag = _tensor_diagnostics(transformed[2], w)
    output = []

    try:
        reference_base = _posterior_oracle(*base)
        reference_transported = _posterior_oracle(*transformed)
        ref_base = _raw_reference(*reference_base)
        ref_transported = _raw_reference(*reference_transported)
        input_roundoff = _error(ref_transported, ref_base, tol)
    except Exception as exc:
        return [{
            "precision": precision, "dim": dim, "kappa_s": kappa_s,
            "kappa_w": kappa_w, "seed": seed, "safe_gate": safe,
            "state": "oracle_error", "exception": repr(exc),
            "base_input_diagnostics": bdiag,
            "transport_input_diagnostics": tdiag,
        }]

    for whiten in MODES:
        row = {
            "precision": precision, "dim": dim, "kappa_s": kappa_s,
            "kappa_w": kappa_w, "seed": seed,
            "whiten_interference": whiten,
            "safe_gate": safe,
            "base_input_diagnostics": bdiag,
            "transport_input_diagnostics": tdiag,
            "original_oracle": ref_base,
            "transported_oracle": ref_transported,
            "input_transport_roundoff": input_roundoff,
        }
        try:
            xb, vb = lmmse_equalizer(*base, whiten_interference=whiten,
                                     precision=precision)
            xt, vt = lmmse_equalizer(
                *transformed, whiten_interference=whiten,
                precision=precision
            )
            if not (torch.isfinite(xb).all() and torch.isfinite(vb).all()
                    and torch.isfinite(xt).all() and torch.isfinite(vt).all()):
                raise FloatingPointError("nonfinite output")
            rb, rt = _raw_output(xb, vb), _raw_output(xt, vt)
            row["original_output"] = rb
            row["transported_output"] = rt
            row["original_solver_error"] = _error(rb, ref_base, tol)
            row["transported_solver_error"] = _error(rt, ref_transported, tol)
            row["end_to_end_frame_error"] = _error(rt, rb, tol)
            row["state"] = "ok"
        except Exception as exc:
            row["state"] = "solver_error"
            row["exception"] = repr(exc)
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Path to full raw JSON")
    args = parser.parse_args()
    torch.set_num_threads(1)

    rows = []
    for precision in PRECISIONS:
        for dim in NUM_RX:
            for condition_s in COV_CONDITIONS:
                for condition_w in FRAME_CONDITIONS:
                    for seed in SEEDS:
                        rows.extend(_trial(
                            precision, dim, condition_s, condition_w, seed
                        ))

    safe_rows = [r for r in rows if r["safe_gate"]]
    good_rows = [r for r in rows if r["state"] == "ok"]
    safe_failed = [
        r for r in safe_rows
        if r["state"] != "ok"
        or r["original_solver_error"]["max_tolerance_ratio"] > 1
        or r["transported_solver_error"]["max_tolerance_ratio"] > 1
        or r["end_to_end_frame_error"]["max_tolerance_ratio"] > 1
    ]
    failures = [r for r in rows if r["state"] != "ok"]
    fields = ["original_solver_error", "transported_solver_error",
              "end_to_end_frame_error", "input_transport_roundoff"]
    summary = {
        "tested_rows": len(rows),
        "successful_rows": len(good_rows),
        "safe_rows": len(safe_rows),
        "safe_gate_failed_rows": len(safe_failed),
        "failed_rows": len(failures),
        "safe_acceptance": (
            "max per-component error <= 2e-4(single) or 2e-11(double) "
            "times (1+abs(reference)); for original, transported, "
            "and end-to-end outputs"
        ),
        "fields": {
            key: {
                "max_abs_x": max(
                    (r[key]["max_abs_x"] for r in good_rows), default=None
                ),
                "max_abs_no_eff": max(
                    (r[key]["max_abs_no_eff"] for r in good_rows),
                    default=None
                ),
                "max_tolerance_ratio": max(
                    (r[key]["max_tolerance_ratio"] for r in good_rows),
                    default=None
                ),
            }
            for key in fields
        },
        "safe_failures_first_12": [
            {k: r.get(k) for k in (
                "precision", "dim", "kappa_s", "kappa_w", "seed",
                "whiten_interference", "state", "exception",
                "original_solver_error", "transported_solver_error",
                "end_to_end_frame_error")}
            for r in safe_failed[:12]
        ],
        "worst_transported_solver_12": [
            {k: r.get(k) for k in (
                "precision", "dim", "kappa_s", "kappa_w", "seed",
                "whiten_interference", "transported_solver_error",
                "input_transport_roundoff")}
            for r in sorted(
                good_rows,
                key=lambda x: x["transported_solver_error"][
                    "max_tolerance_ratio"], reverse=True
            )[:12]
        ],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({
            "protocol": {
                "seed": 20261117,
                "covariance_conditions": COV_CONDITIONS,
                "frame_conditions": FRAME_CONDITIONS,
                "seeds": SEEDS,
                "dims": NUM_RX,
                "precisions": PRECISIONS,
                "modes": MODES,
                "oracle_digits": 100,
                "pytorch": torch.__version__,
                "device": "CPU",
                "safe_gate_rule": summary["safe_acceptance"],
            },
            "summary": summary,
            "raw_rows": rows,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True, indent=2))
    if safe_failed:
        raise SystemExit(
            f"Preregistered SAFE gate failed in {len(safe_failed)} rows; "
            "see raw artifact for unfiltered evidence."
        )


if __name__ == "__main__":
    main()
