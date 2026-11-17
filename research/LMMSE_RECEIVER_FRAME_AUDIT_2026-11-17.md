# Receiver-frame invariance of Sionna PHY LMMSE — research audit

Date: 2026-11-17
Target: Sionna PHY 2.2.0, fork base commit `6237a8dad8635c49f51e9b4c1df95a78899f0d72`.
Isolated branch: `research/lmmse-receiver-frame-invariance-2026-11-17`.

## Verdict

The analytic LMMSE sufficient statistics are invariant to a coherent invertible
receiver-coordinate transformation. The current Sionna implementation preserved
the tested symbol estimates and effective-noise variances in **8 of 8** targeted
cases on each of two supported CPU runtimes. This result complements, but is
**not part of**, the submitted NVIDIA PHY PR #1208, which currently covers
`whiten_channel` and `MaximumLikelihoodDetector`.

## Exact mathematical claim [P, with assumptions A]

Assume (S=S^\dagger\succ0), (W) invertible,
(y'=Wy), (H'=WH), and (S'=WSW^\dagger).
Since (S'^{-1}=(W^\dagger)^{-1}S^{-1}W^{-1}),

\[
Q'=H'^\dagger S'^{-1}H'=H^\dagger S^{-1}H=Q,
\qquad
b'=H'^\dagger S'^{-1}y'=H^\dagger S^{-1}y=b.
\]

With unit transmit-symbol covariance, let (C=(I+Q)^{-1}).
The unscaled posterior mean is (Cb).
For (d=\operatorname{diag}(I-C)), Sionna's streamwise
equalizer convention is

\[
\widehat x=(Cb)\oslash d,\qquad
n_{\rm eff}=\operatorname{Re}(1\oslash d-1).
\]

Both are invariant in exact arithmetic.
A nonunitary (W) **does not** preserve Euclidean distances; the proof
requires transporting (S), not just (y) and (H).

## Live implementation tested [M]

- Production function: `sionna.phy.mimo.equalization.lmmse_equalizer`.
- Both `whiten_interference=True` and `False`.
- (M\in\{2,4\}), two streams, batch size four, synthetic colored
  Hermitian-positive-definite covariance.
- Single and double complex precision.
- Two nonunitary receiver-frame transformations for each test.
- Independent computational oracle built from direct covariance solves and
  (C=(I+H^\dagger S^{-1}H)^{-1}); neither the Sionna LMMSE implementation
  nor the whitening function is reused to construct that oracle.
- Negative control deliberately transports (y,H) but not (S); the test
  rejects an apparent invariance of the resulting *different noise model*.

Numerical witness: [GitHub Actions run 44041648016](https://github.com/jesusvilela/sionna/actions/runs/44041648016).

| Runtime | Research tests | Existing LMMSE smoke tests |
|---|---:|---:|
| Python 3.11 / PyTorch 2.9.1 CPU | 8 passed | 6 passed, 56 deselected |
| Python 3.13 / PyTorch 2.10.0 CPU | 8 passed | 6 passed, 56 deselected |

The test suite uses (\mathrm{atol}=\mathrm{rtol}=2\cdot10^{-4})
(single) or (2\cdot10^{-11}) (double). These are finite tolerance
checks, not exact equality in floating-point arithmetic.

## Strength and limitations

**Strongest surviving result:** exact algebraic invariance of the
LMMSE sufficient statistics, plus software confirmation that
the tested Sionna implementation agrees with an independently formulated
posterior-covariance solve across two runtime versions and two whitening
configurations. A metamorphic negative control checks the direction of
the covariance transformation.

**Most damaging obstruction to promotion:** no CUDA/backend runs,
randomized condition-number sweep, realistic receiver trace,
independent hardware implementation, or end-to-end SER test.
The independent oracle still shares PyTorch linear-algebra kernels with
the implementation under test, so the two paths are algebraically
independent but not independent implementations.

**Resource parity:** no performance improvement is claimed or measured.
The separate adaptive power-of-two / componentwise-divide experiments
showed unacceptable ordinary-input CPU overhead and are not candidates
for inclusion in this contribution.

## Evidence ledger

| Claim | Classification |
|---|---|
| Invariance under invertible coherent receiver-frame transport | P (conditional on SPD (S)) |
| Fixed seeded/synthetic CPU tests agree for the tested cases | M |
| Numerical agreement extends to arbitrary matrices, hardware, or compilers | H, unverified |
| Improved physical 6G signal quality or BER/SER | H, unverified |
| Existing adaptive-normalization variants should enter production | R, retired on measured CPU cost |

## Decision, falsifier and stop criterion

Keep this as a **separate research branch**, without modifying NVIDIA
PR #1208 or opening another upstream PR before maintainer review.

Bounded future experiment, if requested: compare both LMMSE paths
against the direct posterior oracle on randomly generated well-conditioned
complex SPD covariance matrices and invertible frames at varying condition
numbers. The primary metric would be absolute/relative error in
`x_hat` and `no_eff` versus the oracle, with CPU runtime held fixed
and true counterexamples logged without clipping. **Kill the broad
numerical-robustness interpretation** if the errors exceed the published
tolerances on supported dtypes. **Stop altogether** if no upstream
reviewer, actual deployment defect, or independent hardware trace
motivates the work.

## Files

- [Independent experiment](https://github.com/jesusvilela/sionna/blob/research/lmmse-receiver-frame-invariance-2026-11-17/research/tests/test_lmmse_receiver_frame.py)
- [Isolated CPU matrix workflow](https://github.com/jesusvilela/sionna/blob/research/lmmse-receiver-frame-invariance-2026-11-17/.github/workflows/research-lmmse-receiver-frame.yml)
- [Actions evidence](https://github.com/jesusvilela/sionna/actions/runs/44041648016)
- [Upstream NVIDIA PHY contribution](https://github.com/NVlabs/sionna/pull/1208)
