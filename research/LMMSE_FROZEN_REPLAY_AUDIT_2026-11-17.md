# LMMSE frozen-input replay — corrected reproducibility audit

**Date:** 17 November 2026  
**Scope:** Sionna PHY, CPU LMMSE equalization, synthetic colored covariances, exact serialized inputs  
**Status:** research-only. No modification to NVIDIA Sionna PHY PR #1208 or Sionna RT PR #82.

## Verdict

Six original/transported tensor cases were frozen once as exact interleaved
little-endian complex bytes, committed with per-tensor SHA-256 checksums, then
replayed three times under Python 3.11 / PyTorch 2.9.1 and Python 3.13 /
PyTorch 2.10.0, in **two separate GitHub Actions runs**. The exact input
fixture and all 12 fixture/mode **failure classifications** matched in every
comparison: nine finite combinations and three nonfinite whitening
combinations. All moderate-condition controls passed. However, finite output
**bits were not universally reproducible**. In the first run, the nine
finite pairs happened to match across versions; in the second, none did.
Within a runtime/job, all outputs stayed bitwise stable across the three
immediate repeats. In particular, Python 3.11 / PyTorch 2.9.1 changed
finite output bits across the two runs while Python 3.13 / PyTorch 2.10.0
did not. This is evidence of a runtime/environment-sensitive numerical
path, **not** a proved PyTorch-version bug or an effect of receiver physics.

## Frozen object and exact provenance

**Fixture:** [research/fixtures/lmmse_tensor_replay_v1.json](fixtures/lmmse_tensor_replay_v1.json)

Committed JSON file SHA-256:

\`f6a943a80f49286de247f8327e620f2155086fda88210acfcef9863951b76995\`

The original generator used Python 3.13.18 and PyTorch 2.10.0+cpu on
GitHub-hosted x86_64 Linux. It imported the historical
\`lmmse_conditioning_sweep._case\`, whose QR and random draws are
**not** intrinsically cross-version bit-stable. The fixture stores:

- Original receiver observation, channel and covariance: \`y\`, \`h\`, \`s\`.
- Precomputed *transported* tensors: \`yp\`, \`hp\`, \`sp\`.
- Frame matrix \`w\`.
- Dtype, shape, byte-exact \`raw_le_hex\` and SHA-256 for **each tensor**.

Replay deserializes these as raw little-endian complex64/complex128.
It verifies each SHA and exact round-trip bytes and **does not
regenerate QR/random data or reconstruct transported tensors for solver
inputs**. Recomputation of \(WSW^\dagger\) is performed only as an
independent diagnostic.

The six selected cases comprise a 2x2 complex64 moderate-condition control,
2x2 complex64 historical extreme cases with receiver-frame target
condition 30 and 300, a 4x4 complex64 historical extreme case, a
2x2 complex128 moderate control, and a 2x2 complex128 extreme control.
Selection is **purposive**, not a representative distribution of radio
channels.

## Exact question

Can variation in the original stress sweep be explained entirely by
the fact that its seeded matrix generation, QR, and matrix products did
not freeze exact floating-point input bytes across runs?

**Answer:** not entirely. Fixed inputs stabilize failure **classification**
for these six cases, but they do not universally stabilize finite output
bits across hosted runs. That invalidates both the blanket
"seeded generator is sufficient" claim and the stronger
"frozen input guarantees bitwise reproducibility across arbitrary
CPU hosts/libraries" claim.

## Results

The following counts are **per environment and per CI run**; each has
six fixtures × two whitening modes × three repeats = **36 replay rows**.

| Metric | Python 3.11 / Torch 2.9.1 | Python 3.13 / Torch 2.10.0 |
|---|---:|---:|
| Moderate (safe) rows passing | 12/12 | 12/12 |
| Nonfinite extreme rows | 9/36 | 9/36 |
| Finite extreme out-of-tolerance rows: first run | 9 | 9 |
| Finite extreme out-of-tolerance rows: second run | 6 | 9 |
| Within-job bitwise repeats | all 12 configurations | all 12 configurations |
| Between-run finite bitwise pairs unchanged | **0/9** | **9/9** |

**Cross-version within-run comparison:** 9/9 finite pairs identical
(original and transported result SHA-256) in the **first** run; 0/9 in
the **second** run. The three nonfinite configuration states agreed
in both runs.

In both versions and both runs, source covariance Cholesky factorization
passed, while the three high-condition float32 transported covariances
returned \`torch.linalg.cholesky_ex\` info **2, 2, 3**. The failing
states occurred with \`whiten_interference=True\`; corresponding direct
LMMSE cases returned finite values, although they were not always
accurate within the deliberately stringent stress tolerance.

**Direct matched numeric differences between run 1 and run 2
at fixed input bytes (Python 3.11, repeat 0):**

| Case / mode | Largest original-output absolute difference | Largest transported-output absolute difference |
|---|---:|---:|
| 2x2 safe float32, no whitening | \(1.33\times 10^{-7}\) | \(2.38\times 10^{-7}\) |
| 2x2 safe float32, whitening | \(9.54\times 10^{-7}\) | \(3.63\times 10^{-7}\) |
| 2x2 extreme float32, frame target 30, no whitening | \(9.54\times 10^{-7}\) | \(3.34\times 10^{-6}\) |
| 2x2 extreme float32, frame target 300, no whitening | \(7.15\times 10^{-7}\) | \(1.53\times 10^{-5}\) |
| 4x4 extreme float32, frame target 300, no whitening | \(1.33\times 10^{-6}\) | **\(1.41\times 10^{-3}\)** |
| 2x2 safe float64, no whitening | \(4.44\times 10^{-16}\) | \(2.22\times 10^{-16}\) |
| 2x2 extreme float64, whitening | \(6.94\times 10^{-11}\) | \(9.93\times 10^{-11}\) |

These are absolute numerical differences in outputs, **not** BER, SER,
or percentages. The Python 3.13 finite outputs, by contrast, were
bitwise identical between these two runs.

## Representation attack: recompute, but do not replace, the frozen covariance

The second replay adds a diagnostic computing \`w @ s @ w.mH\`
from the same frozen source tensors, without changing the primary
\`sp\` passed to Sionna.

- Python 3.13 / Torch 2.10.0: **6/6** recomputed
  transformed covariances matched the frozen \`sp\` *bit-for-bit*.
- Python 3.11 / Torch 2.9.1: **0/6** matched bit-for-bit.
  Differences were small but nonzero:
  \(2.98\times10^{-8}\) at the 2x2 float32 moderate control,
  \(1.49\times10^{-8}\) at the 4x4 float32 extreme case,
  and approximately \(10^{-17}\) for the float64 examples.

Therefore, even with fixed represented input arrays, the **floating-point
evaluation of an equivalent algebraic construction can differ across
runtime/host configurations**. It is crucial that the primary replay
used the *stored* \`sp\`, not the recomputed one.

This points to a library/backend/CPU dispatch factor, but cannot
distinguish PyTorch-version effects from GitHub runner CPU ISA,
BLAS/LAPACK dispatch, or other host differences. The current runner
metadata records \`x86_64\`, but not a model/ISA fingerprint; on
these jobs \`platform.processor()\` was empty.

## Independent witness, controls, and proof boundary

- **Exact [P]:** If \(S\succ0\), \(W\) invertible, and
  \(S'=WSW^\dagger\), then
  \(H'^\dagger S'^{-1}H'=H^\dagger S^{-1}H\) and
  \(H'^\dagger S'^{-1}y'=H^\dagger S^{-1}y\).
- **Measured [M]:** SHA-verified byte identity of inputs in two runs
  and two CPU runtimes; same failure classification in all 12
  fixture/mode combinations; safe controls pass;
  finite output bits differ in some independent runs.
- **Assumptions [A]:** little-endian supported complex tensor layout,
  fixed committed test data, CPU single-thread operations, and
  positive-definite covariance for the exact LMMSE theorem.
- **Hypotheses [H]:** hardware ISA or backend dispatch explains the
  inter-run bit differences; no controlled hardware parity experiment
  has isolated a cause.
- **Semantic [S]:** a receiver "moving frame" is a coordinate
  representation change, not an altered propagation law.
- **Retired [R]:** "the old variation is only from QR/random input
  generation"; "fixed input guarantees output bits across any supported
  CPU run"; "a finite-precision Cholesky failure disproves the
  exact SPD-coordinate theorem".

The independent numerical oracle uses 100-digit **mpmath** on the
actual stored complex tensors and the posterior-coordinate calculation,
not Sionna's equalizer/whitening implementation. This is an independent
arithmetic backend, **not independent physical measurement**.

The moderate-control acceptance threshold was declared in the replay
before seeing its results: \(2\times10^{-4}\) scaled single-precision
tolerance, \(2\times10^{-11}\) scaled double-precision tolerance.
Both safe control cases × two whitening modes × three repetitions
passed per environment in both runs. The extreme cases were recorded,
not filtered or forcibly clipped into the accepted region.

## Specification → implementation → observation → interpretation

The intended claim was exact-source/transport representation
invariance. The fixture is explicitly byte-frozen, and the
implementation checks bytes at deserialization. The tests compare
LMMSE outputs and posterior-mpmath observables. The measured result
supports invariant **failure classes** on the selected examples,
but does not support portable bitwise agreement across runs. The
original claim is therefore narrowed, with the numerical discrepancies
preserved rather than described as proof failure or physical benefit.

The main unresolved causal question is whether the altered 2.9.1 CPU
finite outputs were caused by hardware model/ISA, PyTorch's
vendor-library selection, or nondeterminism from another source. The
first vs second runs used the **same PyTorch version** within each
environment, so "PyTorch version alone" cannot account for that
inter-run change.

## Exact evidence and source artifacts

- [Frozen inputs](fixtures/lmmse_tensor_replay_v1.json)
- [One-time fixture generator](experiments/freeze_lmmse_cases.py)
- [Replay evaluator](experiments/replay_lmmse_cases.py)
- [Preregistered claim/evidence manifest](LMMSE_FROZEN_REPLAY_MANIFEST_2026-11-17.json)
- [First replay — run 44048790316](https://github.com/jesusvilela/sionna/actions/runs/44048790316)
- [Second replay with representation diagnostics — run 44049760075](https://github.com/jesusvilela/sionna/actions/runs/44049760075)

Each replay run has two attached 30-day raw JSON artifacts, one per
Python/Torch environment. The frozen fixture itself is committed
permanently to this research branch.

## Next bounded experiment and stop rule

The next possible experiment is a **single matched hardware/ISA replay**
of the same fixture under both PyTorch builds, with \`/proc/cpuinfo\`
model, supported instruction flags, linked BLAS configuration and
Torch backend selections recorded. Test the source and transported
outputs and the covariance matmul construction independently. **Kill
a PyTorch-specific causal explanation** unless differences remain on
the same hardware with matched vendor-library settings and
compiler/ISA controls. **Stop** without this experiment if upstream
maintainers report no practical receiver defect and no real input
trace exhibits the extreme-scale problem.

**Decision:** Do not open an upstream issue or modify
[NVIDIA PHY PR #1208](https://github.com/NVlabs/sionna/pull/1208)
or [Sionna RT PR #82](https://github.com/NVlabs/sionna-rt/pull/82).
The frozen replay has served its audit purpose; production changes
would require realistic data and an independently reproduced defect.
