# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Research-only independent receiver-frame checks for the Sionna LMMSE equalizer.

This is an isolated experiment, not an upstream contribution or production fix.
The independent oracle uses direct covariance solves and the posterior matrix,
not Sionna's equalizer/whitening implementations.
"""

import math

import pytest
import torch

from sionna.phy.mimo.equalization import lmmse_equalizer


def _case(precision: str, num_rx: int):
    rdtype = torch.float32 if precision == "single" else torch.float64
    cdtype = torch.complex64 if precision == "single" else torch.complex128
    batch_size = 4
    streams = 2

    real = torch.arange(
        1, batch_size * num_rx * streams + 1, dtype=rdtype
    ).reshape(batch_size, num_rx, streams)
    h = torch.complex(real / (5 * num_rx), torch.flip(real, [1]) / (7 * num_rx))
    h = h.to(cdtype)

    x = torch.tensor(
        [[1 + 1j, -1 + 1j], [-1 - 1j, 1 - 1j],
         [1 - 1j, 1 + 1j], [-1 + 1j, -1 - 1j]],
        dtype=cdtype,
    ) / math.sqrt(2.0)

    noise = torch.complex(
        torch.linspace(0.06, 0.16, batch_size * num_rx, dtype=rdtype),
        torch.linspace(-0.07, 0.13, batch_size * num_rx, dtype=rdtype),
    ).reshape(batch_size, num_rx)

    y = (h @ x.unsqueeze(-1)).squeeze(-1) + noise

    c = torch.eye(num_rx, dtype=cdtype)
    c[0, 1] = 0.26 + 0.09j
    c[-1, 0] = -0.14 + 0.05j
    s0 = c @ c.mH + 0.65 * torch.eye(num_rx, dtype=cdtype)
    s = s0.expand(batch_size, -1, -1).clone()

    w = torch.diag(
        torch.linspace(0.7, 1.45, num_rx, dtype=rdtype)
    ).to(cdtype)
    w[0, 1] = 0.15 + 0.07j
    w[-1, 0] = -0.13 + 0.05j

    w2 = torch.eye(num_rx, dtype=cdtype)
    w2[0, -1] = 0.08 - 0.12j
    w2[-1, -1] = 1.1 + 0.04j
    return y, h, s, (w, w2 @ w)


def _independent_lmmse_oracle(y, h, s):
    """Posterior-coordinate oracle using independent covariance solves.

    A = I + H^H S^{-1} H ; C = A^{-1}
    m = C H^H S^{-1} y ; d = diag(I-C)
    x_hat = m/d ; no_eff = Re(1/d - 1).
    """
    streams = h.shape[-1]
    eye = torch.eye(streams, dtype=h.dtype, device=h.device)
    gram = h.mH @ torch.linalg.solve(s, h)
    a = eye + gram
    posterior_cov = torch.linalg.solve(a, eye)
    weighted_y = torch.linalg.solve(s, y.unsqueeze(-1))
    posterior_mean = (
        posterior_cov @ h.mH @ weighted_y
    ).squeeze(-1)
    diagonal = 1.0 - torch.diagonal(posterior_cov, dim1=-2, dim2=-1)
    return posterior_mean / diagonal, (1.0 / diagonal - 1.0).real


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("num_rx", [2, 4])
@pytest.mark.parametrize("whiten_interference", [True, False])
def test_lmmse_receiver_frame_and_posterior_oracle(
    precision, num_rx, whiten_interference
):
    """Whitened and nonwhitened LMMSE must respect nonunitary receiver frames."""
    y, h, s, transforms = _case(precision, num_rx)
    atol = 2e-4 if precision == "single" else 2e-11
    rtol = 2e-4 if precision == "single" else 2e-11

    reference_x, reference_v = _independent_lmmse_oracle(y, h, s)
    base_x, base_v = lmmse_equalizer(
        y, h, s,
        whiten_interference=whiten_interference,
        precision=precision,
    )
    assert torch.isfinite(base_x).all() and torch.isfinite(base_v).all()
    torch.testing.assert_close(base_x, reference_x, atol=atol, rtol=rtol)
    torch.testing.assert_close(base_v, reference_v, atol=atol, rtol=rtol)

    for transform in transforms:
        yp = (transform @ y.unsqueeze(-1)).squeeze(-1)
        hp = transform @ h
        sp = transform @ s @ transform.mH
        x, v = lmmse_equalizer(
            yp, hp, sp,
            whiten_interference=whiten_interference,
            precision=precision,
        )
        ox, ov = _independent_lmmse_oracle(yp, hp, sp)
        assert torch.isfinite(x).all() and torch.isfinite(v).all()
        torch.testing.assert_close(x, ox, atol=atol, rtol=rtol)
        torch.testing.assert_close(v, ov, atol=atol, rtol=rtol)
        torch.testing.assert_close(x, base_x, atol=atol, rtol=rtol)
        torch.testing.assert_close(v, base_v, atol=atol, rtol=rtol)

    # Deliberately omit covariance transport. This changes the noise model.
    w = transforms[0]
    wrong_x, wrong_v = lmmse_equalizer(
        (w @ y.unsqueeze(-1)).squeeze(-1), w @ h, s,
        whiten_interference=whiten_interference,
        precision=precision,
    )
    discrepancy = torch.maximum(
        (wrong_x - base_x).abs().max(), (wrong_v - base_v).abs().max()
    )
    assert discrepancy > 1e-3
