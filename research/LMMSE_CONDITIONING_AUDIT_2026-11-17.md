# LMMSE receiver-frame conditioning — adversarial audit

**Date:** 17 November 2026  
**Repository:** jesusvilela/sionna, research/lmmse-conditioning-audit-2026-11-17  
**Core experiment:** research/experiments/lmmse_conditioning_sweep.py  
**Latest full-output evidence:** [GitHub Actions run 44046202194](https://github.com/jesusvilela/sionna/actions/runs/44046202194), with separately retained Python 3.11 and Python 3.13 JSON artifacts.  
**Earlier reproductions:** [run 44045467845](https://github.com/jesusvilela/sionna/actions/runs/44045467845) and [run 44044950856](https://github.com/jesusvilela/sionna/actions/runs/44044950856).

## Verdict

The algebraic LMMSE receiver-frame invariance is correct under the
positive-definite covariance and invertibility assumptions, and **all 64
preregistered moderately conditioned tests per CPU environment passed**.
It is **not uniformly reliable in finite precision** for strongly
conditioned transformations: between 36 and 38 otherwise completed
extreme-stress cases exceeded the defined numerical tolerance per run,
and between two and four single-precision whitening rows per run produced
nonfinite outputs. In the nonfinite cases, the *transported floating-point
covariance itself* failed the numerical positive-definiteness check. These are adverse numerical
results, not counterexamples to the algebraic theorem. They do not
establish a new Sionna product bug or a physical 6G effect.

## Claim and evidence ledger

| Claim | Before | Artifact | Audit result | After |
|---|---|---|---|---|
| Coherent invertible receiver-frame transport preserves LMMSE statistics | P, conditional | Algebra below | Survives | P |
| Moderate covariance and frame conditioning preserve outputs numerically | H | 64 safe cases on two CPUs | No safe failures; independent oracle agreement | M, tested regime only |
| Every invertible receiver-frame change is safe in single precision | H | 256-case sweeps per environment | Invalid: some transported tensors lost numerical SPD | R |
| Solver error and input-transport rounding can be distinguished | H | Dual 100-digit actual-input oracles | Recorded as different observables | M, sampled |
| Unconditional adaptive normalization improves production | H | Earlier matched CPU experiments | Too much ordinary-case overhead | R |
| A particular PyTorch version causes these failures | H | Unmatched hosted-run observations | Not established | H |
| Receiver SER, BER or sensitivity is improved | H | No RF data | Not established | H |

## Strongest surviving result

For an invertible complex receiver-coordinate matrix \(W\), Hermitian
positive-definite noise covariance \(S\), coherently transformed
\(y'=Wy\), \(H'=WH\), and \(S'=WSW^\dagger\), define

\[
Q=H^\dagger S^{-1}H,\qquad b=H^\dagger S^{-1}y.
\]

Then \(Q'=Q\) and \(b'=b\). Consequently the LMMSE posterior
\(C=(I+Q)^{-1}\), the posterior mean \(Cb\), and the Sionna streamwise
outputs are invariant **in exact arithmetic**, provided the normalization
\(d=\operatorname{diag}(I-C)\) has no zero entries.

The normal/moderate numerical evidence was explicitly gated in advance:

- Target covariance condition numbers 1 and 30.
- Target receiver-frame condition numbers 1 and 3.
- Dimensions 2 and 4, two streams, two fixed seeds.
- Single and double precision, whitening on and off.
- **64 passing rows per Python/PyTorch environment**, 128 passing rows total.
- Pass tolerance \(2\times10^{-4}(1+|reference|)\) for single and
  \(2\times10^{-11}(1+|reference|)\) for double, assessed separately for
  original-frame solver error, transformed-frame solver error, and total
  end-to-end frame discrepancy.

The largest observed safe-case tolerance ratio in the final run was
approximately **0.005862** (single) and **0.0001031** (double), well below
the required ratio of 1. These are empirical results, not global error
bounds.

## Load-bearing numerical obstruction

The wider matrix grid covered target \(\kappa_2(S)\) in
\(\{1,30,10^3,10^6\}\) and target \(\kappa_2(W)\) in
\(\{1,3,30,300\}\), with 2/4 receive dimensions, two seeded samples,
two precisions and both whitening settings: **256 rows per environment**.

| Final CPU environment | Safe failures / 64 | Completed rows / 256 | Nonfinite solver rows | Out-of-tolerance completed rows outside safe gate |
|---|---:|---:|---:|---:|
| Python 3.11, PyTorch 2.9.1 | 0 | 254 | 2 | 38 |
| Python 3.13, PyTorch 2.10.0 | 0 | 252 | 4 | 36 |

These are counts from the **latest exact-run 44046202194**, not a
global failure rate. An earlier run 44045467845 reported 252 completed,
four nonfinite and 36 out-of-tolerance rows in **both** runtimes.
Run 44044950856 instead reported two nonfinite in Python 3.11.
Thus counts vary in the high-stress region even with seeded scripts.
The total problematic count in these runs remained 40 rows per runtime,
but some rows changed between nonfinite and finite inaccurate output.

The nonfinite rows involved **single precision,
whiten_interference=True, target \(\kappa_2(S)=10^6\)** and a frame
condition target of 30 or 300. Recorded
torch.linalg.cholesky_ex diagnostics returned **info 2 or 3** for the
transported covariance, while the original covariance returned info 0.

For example, one 2-by-2 transformed tensor had measured
\(\kappa_2(S')\) near \(1.23\times10^8\), with a nonzero
Hermitian-roundoff residual and Cholesky info 2. The underlying exact
mathematical covariance must remain positive definite. But finite-
precision matrix multiplication can destroy that property.

The relevant bound is

\[
\kappa_2(WSW^\dagger)\leq
  \kappa_2(W)^2\,\kappa_2(S).
\]

At target \(\kappa_2(S)=10^6\) and \(\kappa_2(W)=300\), the
upper bound reaches \(9\times10^{10}\), far beyond a comfortable
float32 accuracy regime. This is a conditioning argument, **not**
proof that a specific backend must fail.

Completed extreme cases also showed large numerical errors. In the
first Python 3.11 run, one finite whitening output deviated by more
than \(10^5\) in effective-noise variance against the 100-digit oracle.
That giant discrepancy was not reproduced as the same finite output in
the later run; the latter classified the corresponding extreme input
among nonfinite results. We preserve both results rather than
attributing the difference to PyTorch without checking exact fixture
identity.

## Specification -> implementation -> test audit

The specification compares the same physical linear-Gaussian model
under coherent receiver-coordinate changes.

The implementation calls the public Sionna lmmse_equalizer in both
whitening modes. A separate oracle uses **100-digit mpmath** to invert
the **actual represented** input tensors; it does not call the Sionna
LMMSE or whitening functions. The test records:

1. Source-frame solver error against its represented-input oracle.
2. Transformed-frame solver error against its represented-input oracle.
3. Difference between the two arbitrary-precision oracles, quantifying
   *input representation/transport drift*.
4. End-to-end difference between computed LMMSE outputs.
5. Tensor condition estimates, raw Hermiticity residuals and Cholesky
   factorization status.

The raw JSON contains per-case outputs, references, unnormalized
absolute errors, normalized tolerance ratios, matrix diagnostics and
exceptions. Each run preserves two separately labeled artifacts for
Python 3.11 and 3.13.

Crucially, a covariance that has lost numerical positive-definiteness
is **outside the theorem's represented-input assumptions**. It must not
be promoted as a counterexample to the exact theorem.

## Direction, type and metric audit

- The bound on \(\kappa_2(WSW^\dagger)\) is an **upper bound**, not an
  equality or empirical estimate.
- Condition numbers are dimensionless; error in LMMSE estimates and
  error in effective-noise variance remain separate observables.
- The tolerance ratio is computed after preserving raw errors. No
  clipping is performed and no high-condition failures are discarded.
- The phrase "256 cases" refers to rows of the controlled parameter
  grid, not 256 independently observed radio channels.
- The environment-level counts are not statistical confidence bounds
  on arbitrary hardware.

## Controls and resource parity

All modes use the same scripted grids. Both whitening settings are
compared on the same represented input matrices and the same mpmath
oracle. The earlier 8-test LMMSE regression is rerun before each sweep.
It contains a deliberately incorrect covariance-transport negative
control, which must change the observable.

No runtime-performance benefit is claimed; mpmath is intentionally
more expensive and is only a numerical reference.

## Independent-witness status

The oracle is **algebraically independent of the Sionna equalizer
implementation** and uses arbitrary-precision mpmath rather than
PyTorch solves. That is a stronger independent numerical witness than
a second call to torch.linalg.solve.

It is **not independent physical validation**: inputs and their
finite-precision transports are still generated by PyTorch on GitHub
hosted CPUs. The script seeds random draws, but does not currently
persist hashes of the exact original and transported matrix bytes.
Thus differences between separate hosted runs cannot yet be attributed
to a library version or CPU backend. This is unresolved remainder.

Multiple runs disagree in the number of nonfinite cases
for one runtime, including after a documentation-only revision.
That observation explicitly blocks any claim of bitwise cross-run
reproducibility at the most extreme condition settings.

## Round-trip coherence

The exact mathematics establishes frame invariance for invertible
\(W\) and SPD \(S\). The software tests establish finite tolerance
agreement on moderate cases. Stress tests establish that **the
numerically represented transformed covariance** can cease to be SPD,
and the solver may then return nonfinite outputs or large errors.
These are different levels of claim. No numerical observation has
been elevated to a new physical-layer performance result.

## Corrections and retirements

- **Retired:** "any invertible receiver frame is numerically safe".
  The valid claim is exact algebraic invariance conditional on SPD,
  plus measured invariance within the reported moderate regime.
- **Retired:** "all high-conditioning differences are defects in
  Sionna's LMMSE formula". Some inputs violate numerical SPD after
  transformation, so a blanket formula-level conclusion is invalid.
- **Not established:** one PyTorch version is more robust than another.
  Exact fixture hashes and matched hardware are missing.
- **Retained:** the earlier negative result on ordinary-case overhead
  of unconditional adaptive covariance normalization.

## Next bounded experiment and stop rule

- **Object:** exact-byte reproducibility of one extreme witness
  (matrix \(y,H,S,W\)) across the two runtimes and CPU families.
- **Prediction:** for identical serialized inputs, source/transport
  rounding and solver error can be separated more decisively.
- **Primary metric:** measured Hermiticity residual, Cholesky status,
  and actual-input oracle error, with matrix hashes and raw outputs.
- **Controls:** both whitening settings, a moderate-SPD witness and a
  negative covariance-transport witness.
- **Representation attack:** construct \(S'\) in an extended-precision
  reference, then round only once; compare with ordinary float32
  multiplication. Keep the two protocols explicitly distinct.
- **Kill condition:** the supposedly identical input-byte hashes or
  baseline outputs do not match; no cross-version causal inference.
- **Stop condition:** no external reviewer, supported-device defect, or
  deployment trace warrants more work on these extreme synthetic cases.
- **Promotion rule:** only a replicated failure **with represented SPD
  inputs in a supported realistic workload** should motivate an
  upstream numerical improvement or separate PR.

## Final evidence map

- **P:** exact coherent receiver-coordinate covariance identity and
  its LMMSE posterior consequence, under explicit assumptions.
- **A:** invertible \(W\), SPD \(S\), nonzero LMMSE stream normalizers,
  valid linear-Gaussian model.
- **M:** 64/64 safe cases passed per CPU runtime, 36–38 extreme
  out-of-tolerance completed rows and two to four nonfinite rows
  across repeated runs, actual represented-input diagnostics and
  raw JSON artifacts retained separately for each runtime and run.
- **H:** general GPU numerical robustness and physical 6G sensitivity.
- **S:** receiver "moving frame" is a coordinate transformation,
  not a new physical channel.
- **R:** universal numerical invariance and unconditional
  adaptive-normalization deployment claims.

**Decision:** preserve this branch as research and do not modify
[NVIDIA Sionna PHY PR #1208](https://github.com/NVlabs/sionna/pull/1208)
or [Sionna RT PR #82](https://github.com/NVlabs/sionna-rt/pull/82).
