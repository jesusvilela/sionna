"""Tests for the research-only adaptive power-of-two whitening candidate."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from adaptive_whitening import (  # noqa: E402
    adaptive_div_whiten_channel,
    conditional_view_power2_whiten_channel,
    power2_whiten_channel,
)
from sionna.phy.mimo.utils import whiten_channel as baseline_whiten_channel  # noqa: E402


def _base_problem(dtype: torch.dtype, scale: float):
    rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
    s0 = torch.tensor(
        [[1.0 + 0j, 0.25 + 0.1j], [0.25 - 0.1j, 1.5 + 0j]],
        dtype=dtype,
    )
    h0 = torch.linalg.cholesky(s0) @ torch.diag(
        torch.tensor([1.0 + 0j, 0.5 + 0j], dtype=dtype)
    )
    x = torch.tensor([1.0 + 0j, -1j], dtype=dtype)
    noise0 = torch.tensor([0.07 + 0.02j, -0.03 + 0.05j], dtype=dtype)
    root = math.sqrt(scale)
    s = scale * s0
    h = root * h0
    y = h @ x + root * noise0
    return y, h, s, s0, h0, x, noise0


def _mpc(z):
    return mp.mpc(mp.mpf(float(z.real)), mp.mpf(float(z.imag)))


def _mp_matrix(tensor):
    a = tensor.detach().cpu().numpy()
    return mp.matrix([[_mpc(a[i, j]) for j in range(a.shape[1])] for i in range(a.shape[0])])


def _mp_vector(tensor):
    a = tensor.detach().cpu().numpy()
    return mp.matrix([_mpc(a[i]) for i in range(a.shape[0])])


def _oracle_gram(h, s):
    with mp.workdps(100):
        hm = _mp_matrix(h)
        sm = _mp_matrix(s)
        gm = hm.transpose_conj() * (sm ** -1) * hm
        return np.array([[complex(gm[i, j]) for j in range(gm.cols)] for i in range(gm.rows)])


def _oracle_distances(y, h, s, symbols):
    with mp.workdps(100):
        ym = _mp_vector(y)
        hm = _mp_matrix(h)
        sm = _mp_matrix(s)
        values = []
        for i in range(symbols.shape[0]):
            xm = _mp_vector(symbols[i])
            residual = ym - hm * xm
            value = (residual.transpose_conj() * (sm ** -1) * residual)[0]
            values.append(float(mp.re(value)))
        return np.asarray(values)


def _whitened_distances(yw, hw, symbols):
    residual = yw.unsqueeze(0) - symbols @ hw.T
    return residual.abs().square().sum(dim=-1).detach().cpu().numpy()


def test_normal_range_is_exact_noop_against_production():
    y, h, s, *_ = _base_problem(torch.complex128, 1e-200)
    by, bh = baseline_whiten_channel(y, h, s, return_s=False)
    ay, ah, diagnostics = power2_whiten_channel(
        y, h, s, return_diagnostics=True
    )
    assert torch.count_nonzero(diagnostics["applied_exponent"]) == 0
    assert torch.equal(ay, by)
    assert torch.equal(ah, bh)


def test_subnormal_complex128_matches_actual_input_mpmath_oracle():
    y, h, s, _, _, _, _ = _base_problem(torch.complex128, 1e-320)
    symbols = torch.tensor(
        [[1 + 0j, 1j], [-1 + 0j, 1j], [1j, -1j], [-1j, -1 + 0j]],
        dtype=torch.complex128,
    )
    yw, hw, diagnostics = power2_whiten_channel(
        y, h, s, return_diagnostics=True
    )
    assert diagnostics["applied_exponent"].item() != 0
    assert torch.isfinite(yw).all() and torch.isfinite(hw).all()

    gram = (hw.mH @ hw).detach().cpu().numpy()
    gram_ref = _oracle_gram(h, s)
    np.testing.assert_allclose(gram, gram_ref, rtol=0, atol=1e-12)

    distances = _whitened_distances(yw, hw, symbols)
    distances_ref = _oracle_distances(y, h, s, symbols)
    np.testing.assert_allclose(distances, distances_ref, rtol=0, atol=1e-12)


def test_top_exponent_complex128_matches_actual_input_mpmath_oracle():
    y, h, s, _, _, _, _ = _base_problem(torch.complex128, 1e308)
    yw, hw, diagnostics = power2_whiten_channel(
        y, h, s, return_diagnostics=True
    )
    assert diagnostics["applied_exponent"].item() != 0
    assert torch.isfinite(yw).all() and torch.isfinite(hw).all()
    np.testing.assert_allclose(
        (hw.mH @ hw).detach().cpu().numpy(),
        _oracle_gram(h, s),
        rtol=0,
        atol=1e-12,
    )


def test_subnormal_complex64_matches_actual_input_double_solve():
    y, h, s, *_ = _base_problem(torch.complex64, 1e-40)
    yw, hw, diagnostics = power2_whiten_channel(
        y, h, s, return_diagnostics=True
    )
    assert diagnostics["applied_exponent"].item() != 0
    assert torch.isfinite(yw).all() and torch.isfinite(hw).all()

    hd = h.to(torch.complex128)
    sd = s.to(torch.complex128)
    reference = hd.mH @ torch.linalg.solve(sd, hd)
    torch.testing.assert_close(
        hw.to(torch.complex128).mH @ hw.to(torch.complex128),
        reference,
        rtol=1e-5,
        atol=1e-5,
    )


def test_mixed_batch_broadcasts_covariance_exponents():
    y0, h0, s0, *_ = _base_problem(torch.complex128, 1.0)
    scales = (1e-320, 1.0)
    s = torch.stack([s0 * scales[0], s0 * scales[1]]).unsqueeze(1)
    h = torch.stack([h0 * math.sqrt(scales[0]), h0]).unsqueeze(1).expand(-1, 3, -1, -1).clone()
    y = torch.stack([y0 * math.sqrt(scales[0]), y0]).unsqueeze(1).expand(-1, 3, -1).clone()

    yw, hw, diagnostics = power2_whiten_channel(
        y, h, s, return_diagnostics=True
    )
    assert yw.shape == (2, 3, 2)
    assert hw.shape == (2, 3, 2, 2)
    assert diagnostics["applied_exponent"].shape == (2, 1)
    assert diagnostics["applied_exponent"][0, 0] != 0
    assert diagnostics["applied_exponent"][1, 0] == 0
    assert torch.isfinite(yw).all() and torch.isfinite(hw).all()


def test_tensor_only_path_captures_fullgraph_for_normal_and_subnormal():
    fn = lambda yy, hh, ss: power2_whiten_channel(yy, hh, ss)
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    for scale in (1.0, 1e-320):
        y, h, s, *_ = _base_problem(torch.complex128, scale)
        expected = fn(y, h, s)
        actual = compiled(y, h, s)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)



def test_componentwise_division_normal_range_is_exact_noop():
    y, h, s, *_ = _base_problem(torch.complex128, 1e-200)
    by, bh = baseline_whiten_channel(y, h, s, return_s=False)
    ay, ah, diagnostics = adaptive_div_whiten_channel(
        y, h, s, return_diagnostics=True
    )
    assert diagnostics["applied_exponent"].item() == 0
    assert torch.equal(ay, by)
    assert torch.equal(ah, bh)


def test_componentwise_division_subnormal_complex128_matches_mpmath():
    y, h, s, *_ = _base_problem(torch.complex128, 1e-320)
    symbols = torch.tensor(
        [[1 + 0j, 1j], [-1 + 0j, 1j], [1j, -1j], [-1j, -1 + 0j]],
        dtype=torch.complex128,
    )
    yw, hw, diagnostics = adaptive_div_whiten_channel(
        y, h, s, return_diagnostics=True
    )
    assert diagnostics["applied_exponent"].item() != 0
    assert torch.isfinite(yw).all() and torch.isfinite(hw).all()
    np.testing.assert_allclose(
        (hw.mH @ hw).detach().cpu().numpy(),
        _oracle_gram(h, s),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _whitened_distances(yw, hw, symbols),
        _oracle_distances(y, h, s, symbols),
        rtol=0,
        atol=1e-12,
    )


def test_componentwise_division_top_exponent_complex128_matches_mpmath():
    y, h, s, *_ = _base_problem(torch.complex128, 1e308)
    yw, hw, diagnostics = adaptive_div_whiten_channel(
        y, h, s, return_diagnostics=True
    )
    assert diagnostics["applied_exponent"].item() != 0
    assert torch.isfinite(yw).all() and torch.isfinite(hw).all()
    np.testing.assert_allclose(
        (hw.mH @ hw).detach().cpu().numpy(),
        _oracle_gram(h, s),
        rtol=0,
        atol=1e-12,
    )


def test_componentwise_division_subnormal_complex64_matches_double_solve():
    y, h, s, *_ = _base_problem(torch.complex64, 1e-40)
    yw, hw, diagnostics = adaptive_div_whiten_channel(
        y, h, s, return_diagnostics=True
    )
    assert diagnostics["applied_exponent"].item() != 0
    assert torch.isfinite(yw).all() and torch.isfinite(hw).all()
    hd = h.to(torch.complex128)
    sd = s.to(torch.complex128)
    reference = hd.mH @ torch.linalg.solve(sd, hd)
    torch.testing.assert_close(
        hw.to(torch.complex128).mH @ hw.to(torch.complex128),
        reference,
        rtol=1e-5,
        atol=1e-5,
    )


def test_componentwise_division_mixed_batch_broadcasts():
    y0, h0, s0, *_ = _base_problem(torch.complex128, 1.0)
    scales = (1e-320, 1.0)
    s = torch.stack([s0 * scales[0], s0 * scales[1]]).unsqueeze(1)
    h = torch.stack([h0 * math.sqrt(scales[0]), h0]).unsqueeze(1).expand(-1, 3, -1, -1).clone()
    y = torch.stack([y0 * math.sqrt(scales[0]), y0]).unsqueeze(1).expand(-1, 3, -1).clone()
    yw, hw, diagnostics = adaptive_div_whiten_channel(
        y, h, s, return_diagnostics=True
    )
    assert yw.shape == (2, 3, 2)
    assert hw.shape == (2, 3, 2, 2)
    assert diagnostics["applied_exponent"].shape == (2, 1)
    assert diagnostics["applied_exponent"][0, 0] != 0
    assert diagnostics["applied_exponent"][1, 0] == 0
    assert torch.isfinite(yw).all() and torch.isfinite(hw).all()


def test_componentwise_division_captures_fullgraph():
    fn = adaptive_div_whiten_channel
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    for scale in (1.0, 1e-320):
        y, h, s, *_ = _base_problem(torch.complex128, scale)
        expected = fn(y, h, s)
        actual = compiled(y, h, s)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)



def test_conditional_view_path_matches_baseline_and_scaled_candidates():
    normal = _base_problem(torch.complex128, 1.0)
    by, bh = baseline_whiten_channel(normal[0], normal[1], normal[2], return_s=False)
    cy, ch = conditional_view_power2_whiten_channel(
        normal[0], normal[1], normal[2]
    )
    assert torch.equal(cy, by)
    assert torch.equal(ch, bh)

    extreme = _base_problem(torch.complex128, 1e-320)
    cy, ch = conditional_view_power2_whiten_channel(
        extreme[0], extreme[1], extreme[2]
    )
    ay, ah = power2_whiten_channel(extreme[0], extreme[1], extreme[2])
    torch.testing.assert_close(cy, ay, rtol=0, atol=0)
    torch.testing.assert_close(ch, ah, rtol=0, atol=0)


def test_conditional_view_path_captures_fullgraph():
    fn = conditional_view_power2_whiten_channel
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    for scale in (1.0, 1e-320):
        y, h, s, *_ = _base_problem(torch.complex128, scale)
        expected = fn(y, h, s)
        actual = compiled(y, h, s)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
