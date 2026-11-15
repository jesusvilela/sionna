"""Research-only power-of-two adaptive covariance whitening.

The production Sionna implementation is intentionally untouched. This module
compares the existing Cholesky path with a tensor-only scaling candidate:
scale the covariance only when its largest entry is outside the normal
floating-point exponent range, and compensate y/H by the corresponding
half-exponent so the whitened channel is unchanged in exact arithmetic.

This is a numerical-conditioning experiment, not a physical receiver model.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from typing import Any

import torch

from sionna.phy.mimo.utils import whiten_channel as baseline_whiten_channel
from sionna.phy.utils.linalg import inv_cholesky


def _ldexp_tensor(value: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """Apply a binary exponent without feeding subnormals to ldexp directly.

    Some CPU kernels can flush subnormal inputs before applying ldexp. frexp
    first decomposes each component into a normal mantissa and integer exponent;
    ldexp then acts on the mantissa, while exponents are combined exactly.
    """
    if value.is_complex():
        real_mantissa, real_exponent = torch.frexp(value.real)
        imag_mantissa, imag_exponent = torch.frexp(value.imag)
        return torch.complex(
            torch.ldexp(real_mantissa, real_exponent + exponent),
            torch.ldexp(imag_mantissa, imag_exponent + exponent),
        )
    mantissa, value_exponent = torch.frexp(value)
    return torch.ldexp(mantissa, value_exponent + exponent)


def _expand_batch(value: torch.Tensor, batch_shape: tuple[int, ...]) -> torch.Tensor:
    """Left-pad a batch tensor with singleton dimensions and broadcast it."""
    if value.ndim > len(batch_shape):
        raise ValueError("exponent batch rank exceeds output batch rank")
    padded = value.reshape((1,) * (len(batch_shape) - value.ndim) + tuple(value.shape))
    return padded.expand(batch_shape)


def _component_max(s: torch.Tensor) -> torch.Tensor:
    if s.is_complex():
        return torch.maximum(
            torch.amax(torch.abs(s.real), dim=(-2, -1)),
            torch.amax(torch.abs(s.imag), dim=(-2, -1)),
        )
    return torch.amax(torch.abs(s), dim=(-2, -1))


def _exponent_policy(s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw frexp exponents and only the exponents requiring rescaling."""
    real_dtype = s.real.dtype if s.is_complex() else s.dtype
    finfo = torch.finfo(real_dtype)
    lower_normal_exponent = math.frexp(float(finfo.tiny))[1]
    upper_normal_exponent = math.frexp(float(finfo.max))[1] - 1

    max_component = _component_max(s)
    _, exponent = torch.frexp(max_component)
    needs_rescale = (exponent < lower_normal_exponent) | (
        exponent > upper_normal_exponent
    )
    applied = torch.where(needs_rescale, exponent, torch.zeros_like(exponent))
    return exponent, applied


def _divide_by_real_scale(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if value.is_complex():
        return torch.complex(value.real / scale, value.imag / scale)
    return value / scale


def _power2_core(
    y: torch.Tensor,
    h: torch.Tensor,
    s: torch.Tensor,
    applied_exponent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale covariance and data using a preselected exponent."""
    s_unit = _ldexp_tensor(s, -applied_exponent[..., None, None])
    batch_shape = torch.broadcast_shapes(
        tuple(s.shape[:-2]), tuple(h.shape[:-2]), tuple(y.shape[:-1])
    )
    exponent = _expand_batch(applied_exponent, batch_shape)
    half_exp = torch.div(exponent, 2, rounding_mode="floor")
    odd_exp = exponent - 2 * half_exp

    data_exp = -half_exp
    y_scaled = _ldexp_tensor(y, data_exp[..., None])
    h_scaled = _ldexp_tensor(h, data_exp[..., None, None])

    real_dtype = s.real.dtype if s.is_complex() else s.dtype
    inv_sqrt_two = torch.as_tensor(
        1.0 / math.sqrt(2.0), dtype=real_dtype, device=s.device
    )
    one = torch.ones((), dtype=real_dtype, device=s.device)
    odd_factor = torch.where(odd_exp != 0, inv_sqrt_two, one)
    y_scaled = y_scaled * odd_factor[..., None]
    h_scaled = h_scaled * odd_factor[..., None, None]

    l_inv = inv_cholesky(s_unit)
    yw = (l_inv @ y_scaled.unsqueeze(-1)).squeeze(-1)
    hw = l_inv @ h_scaled
    return yw, hw


def power2_whiten_channel(
    y: torch.Tensor,
    h: torch.Tensor,
    s: torch.Tensor,
    *,
    adaptive: bool = True,
    return_s: bool = False,
    return_diagnostics: bool = False,
) -> tuple[Any, ...]:
    """Research candidate using tensor-only binary exponent scaling."""
    raw_exponent, applied_exponent = _exponent_policy(s)
    if not adaptive:
        applied_exponent = raw_exponent

    yw, hw = _power2_core(y, h, s, applied_exponent)
    outputs: tuple[Any, ...]
    if return_s:
        m = s.shape[-1]
        sw = torch.eye(m, dtype=s.dtype, device=s.device)
        if s.shape[:-2]:
            sw = sw.expand(*s.shape[:-2], m, m)
        outputs = (yw, hw, sw)
    else:
        outputs = (yw, hw)

    if return_diagnostics:
        outputs = outputs + ({
            "raw_exponent": raw_exponent,
            "applied_exponent": applied_exponent,
        },)
    return outputs


def adaptive_div_whiten_channel(
    y: torch.Tensor,
    h: torch.Tensor,
    s: torch.Tensor,
    *,
    return_diagnostics: bool = False,
) -> tuple[Any, ...]:
    """Adaptive covariance scaling by componentwise real/imaginary division.

    The scale is the maximum absolute real or imaginary component. It is set
    to one for normal-range covariance inputs, preserving their represented
    values exactly. For extreme inputs, S is divided by scale and y/H by its
    square root. This avoids complex division and avoids ldexp on subnormals.
    """
    raw_exponent, applied_exponent = _exponent_policy(s)
    max_component = _component_max(s)
    scale = torch.where(
        applied_exponent != 0, max_component, torch.ones_like(max_component)
    )
    s_scale = scale[..., None, None]
    s_unit = _divide_by_real_scale(s, s_scale)

    batch_shape = torch.broadcast_shapes(
        tuple(s.shape[:-2]), tuple(h.shape[:-2]), tuple(y.shape[:-1])
    )
    scale_full = _expand_batch(scale, batch_shape)
    root_scale = torch.sqrt(scale_full)
    y_scaled = _divide_by_real_scale(y, root_scale[..., None])
    h_scaled = _divide_by_real_scale(h, root_scale[..., None, None])

    l_inv = inv_cholesky(s_unit)
    yw = (l_inv @ y_scaled.unsqueeze(-1)).squeeze(-1)
    hw = l_inv @ h_scaled

    if return_diagnostics:
        return yw, hw, {
            "raw_exponent": raw_exponent,
            "applied_exponent": applied_exponent,
            "scale": scale,
        }
    return yw, hw


def _view_scaled_branch(y_real, h_real, s_real, applied_exponent):
    y = torch.view_as_complex(y_real)
    h = torch.view_as_complex(h_real)
    s = torch.view_as_complex(s_real)
    return _power2_core(y, h, s, applied_exponent)


def _view_baseline_branch(y_real, h_real, s_real, applied_exponent):
    y = torch.view_as_complex(y_real)
    h = torch.view_as_complex(h_real)
    s = torch.view_as_complex(s_real)
    return baseline_whiten_channel(y, h, s, return_s=False)


def conditional_view_power2_whiten_channel(
    y: torch.Tensor, h: torch.Tensor, s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.cond fast path with complex inputs represented as real views.

    This variant tests whether passing each complex input as one real-valued
    view avoids the real/imaginary input-alias restriction observed with
    torch.cond on complex operands.
    """
    if not (y.is_complex() and h.is_complex() and s.is_complex()):
        raise TypeError("conditional view path currently requires complex tensors")
    _, applied_exponent = _exponent_policy(s)
    predicate = torch.any(applied_exponent != 0)
    operands = (
        torch.view_as_real(y),
        torch.view_as_real(h),
        torch.view_as_real(s),
        applied_exponent,
    )
    return torch.cond(
        predicate,
        _view_scaled_branch,
        _view_baseline_branch,
        operands,
    )


def _benchmark_once(fn, y, h, s, warmup: int, repeats: int, loops: int) -> dict:
    for _ in range(warmup):
        fn(y, h, s)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(loops):
            fn(y, h, s)
        samples.append(1000.0 * (time.perf_counter() - start) / loops)
    return {
        "median_ms_per_call": statistics.median(samples),
        "samples_ms_per_call": samples,
    }


def run_benchmarks() -> dict:
    """Matched CPU timing; informative only, not a universal performance claim."""
    torch.set_num_threads(1)
    result = {}
    for dim in (2, 8):
        for batch in (128, 1024):
            gen = torch.Generator(device="cpu").manual_seed(20261115 + dim + batch)
            c = torch.randn(batch, dim, dim, dtype=torch.complex128, generator=gen)
            s = c @ c.mH + 0.5 * torch.eye(dim, dtype=torch.complex128)
            h = torch.randn(batch, dim, 2, dtype=torch.complex128, generator=gen)
            y = torch.randn(batch, dim, dtype=torch.complex128, generator=gen)
            key = f"n={dim},batch={batch}"
            result[key] = {
                "baseline": _benchmark_once(
                    lambda yy, hh, ss: baseline_whiten_channel(yy, hh, ss, return_s=False), y, h, s, 10, 5, 40
                ),
                "adaptive_power2": _benchmark_once(
                    power2_whiten_channel, y, h, s, 10, 5, 40
                ),
                "adaptive_division": _benchmark_once(
                    adaptive_div_whiten_channel, y, h, s, 10, 5, 40
                ),
                "conditional_view_power2": _benchmark_once(
                    conditional_view_power2_whiten_channel, y, h, s, 10, 5, 40
                ),
                "always_power2": _benchmark_once(
                    lambda yy, hh, ss: power2_whiten_channel(
                        yy, hh, ss, adaptive=False
                    ),
                    y, h, s, 10, 5, 40,
                ),
            }
    return {
        "device": "CPU",
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "timing_scope": "local matched eager calls; not GPU or end-to-end detector latency",
        "results": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.benchmark:
        print(json.dumps(run_benchmarks(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
