# Same-host A-B-A LMMSE replay — evidence audit

**Date:** 17 November 2026  
**Research branch:** research/lmmse-matched-host-aba-2026-11-17  
**Live source commit:** a631ab88f70cffc4be7e67c3f861f618e34fd94b  
**Validated run:** [44054491046](https://github.com/jesusvilela/sionna/actions/runs/44054491046)  
**Result:** three successful sequential runs on one GitHub-hosted VM.

## Verdict

The earlier separate-run discrepancy cannot be treated as an
unavoidable PyTorch-version incompatibility. When both runtimes used
the same six **byte-frozen** source and transported LMMSE cases on the
**same GitHub runner VM**, an A1→B→A2 experiment yielded identical
failure states (36/36 cases per comparison) and identical raw finite
LMMSE output bytes (27/27 finite rows per comparison). The single-
precision covariance Cholesky failures remained reproducible;
normal/moderate control cases passed. Both runtimes also recomputed
the equivalent covariance \`W @ S @ W.mH\` with identical bytes to
each other, though the recomputed bytes **did not match the frozen
transformed covariance generated on the original fixture host**.
These facts implicate an environmental/runtime-dispatch factor as
a plausible explanation for earlier separate-run differences, but do
not identify a unique causal mechanism and do not establish physical
receiver performance.

## Frozen claim and typed observables

**Question under test:** Holding the actual VM CPU environment and
literal floating-point input bytes fixed, do PyTorch 2.9.1 and 2.10.0
produce the same LMMSE outputs on the selected Sionna code path?
Does immediately returning to 2.9.1 reproduce A1's outputs?

The primary observable is a typed tuple:

1. SHA-256 equality of each frozen source/transport tensor.
2. Finite/nonfinite solver classification by fixture and whitening mode.
3. Bytewise equality of finite original and transported output tensors.
4. Separate raw absolute differences for complex symbol estimates and
   real effective-noise variances.
5. Cholesky factorization \`info\` values.
6. Recomputed \`WSW^\dagger\` SHA-256 (diagnostic only, not solver input).
7. Exact 100-digit posterior-oracle comparison for represented inputs.

We do not collapse heterogeneous outputs to one scalar and do not
discard failures to inflate the success rate.

## Host and runtime controls

One GitHub Actions **job**, one runner VM, sequential virtualenv invocations:

| Stage | Interpreter | Torch | Role |
|---|---|---|---|
| A1 | Python 3.11.18 | 2.9.1+cpu | Original runtime |
| B | Python 3.13.18 | 2.10.0+cpu | Alternate runtime |
| A2 | Python 3.11.18 | 2.9.1+cpu | Return-to-original control |

Machine recorded as **Intel Xeon Platinum 8573C**, with reported
AVX-512 CPU capability. Both PyTorch wheels reported **MKL 2024.2**
and **oneDNN (MKL-DNN) 3.7.1**. OpenMP/MKL/OPENBLAS thread count
controlled to one, with dynamic scheduling disabled through
environment variables. Exact \`lscpu\`, \`uname\`, Torch build flags,
source SHA and per-run JSONs are included in the Actions artifact.

The independent environments pinned the same NumPy 2.3.5,
SciPy 1.16.3, h5py 3.15.1, matplotlib 3.10.8,
importlib-resources 6.5.2 and mpmath 1.3.0.

**Residual confound:** The Python interpreter and Torch wheel
versions change together. Although their advertised MKL/oneDNN
versions match, the binary build/compiler flags and possible dispatch
paths remain uncontrolled. A1→B differences, if observed, could
therefore only be attributed to a **runtime-stack** change, not to
PyTorch version alone. In this run, no such finite-output difference
was observed.

## Results: no excluded rows

Six fixed fixtures × two whitening modes × three repetitions =
**36 raw rows per stage**. Three finite-output configurations per
stage became nonfinite only under the specific high-condition
whitening mode, and were represented by nine nonfinite repeated
rows. The remaining nine fixture/mode combinations yielded 27
finite repeated rows.

| Paired comparison | Matching failure classifications | Matching finite original+transported output bytes | Maximum finite estimate error | Maximum finite effective-noise error |
|---|---:|---:|---:|---:|
| A1 vs B | 36/36 | 27/27 | 0 | 0 |
| B vs A2 | 36/36 | 27/27 | 0 | 0 |
| A1 vs A2 | 36/36 | 27/27 | 0 | 0 |

The separately computed 100-digit actual-input posterior reference
was also exactly identical in the exported comparisons. All
12 moderate-control repeated rows in each stage passed the
preregistered precision-dependent tolerance.

**Covariance representation attack:** On the same host,
recomputed \`W @ S @ W.mH\` had the same SHA-256 across all
36 comparisons, even between PyTorch builds. However, for
**all six fixtures**, the recomputed covariance bytes did **not**
match the stored \`sp\` originally produced on a different runner.
This diagnostic did not modify Sionna's solver input: every
primary test used the original byte-frozen \`sp\`.

Three high-condition float32 fixtures retained transported
Cholesky statuses **2, 2, 3** (source statuses 0). Both whitening
modes were preserved in the raw results. We do not interpret
numerically non-SPD represented matrices as counterexamples to the
exact SPD covariance theorem.

## Claim and evidence ledger

| Claim | Before | Witness | Observation | After |
|---|---|---|---|---|
| Coherent invertible transport preserves exact LMMSE statistics | P conditional | Algebraic covariance identity | Unchanged; requires SPD and nonzero stream normalizers | P |
| Frozen cases reproduce high-condition failure classification | M | Earlier two separate runs and same-host A-B-A | All states match | M |
| Different PyTorch releases necessarily return different bits | H | Same-host A1→B→A2 | No difference on this host | R as universal assertion |
| The prior separate-run bit disagreement came only from QR input generation | H | Fixed literal tensor bytes in separate runs | Disagreement persisted there | R |
| The prior separate-run difference came only from PyTorch version | H | Same-host A1→B→A2 | Not supported; B matched A | R |
| Floating-point covariance products can be representation/environment sensitive | H | Freeze vs recompute and cross-run record | Measured on the chosen inputs | M, bounded |
| A specific CPU/BLAS implementation causes the drift | H | Build/host metadata | Not independently isolated | H |
| Real 6G receiver metrics improve | H | No RF traces | Unverified | H |

## Specification → implementation → test → interpretation

**Specification:** Compare semantically identical input states under
two runtime stacks, separate physical host effects, and preserve
the exact output and failure observables.

**Implementation:** The existing Sionna public
\`lmmse_equalizer\` consumes raw deserialized \`y,H,S,y',H',S'\`.
A separate arbitrary-precision posterior oracle is built using
mpmath. The new comparison script checks JSON and input hashes,
compares A1/B/A2 results without clipping, and includes the
return-to-original A2 counterfactual control.

**Verification:** GitHub Actions run 44054491046 completed
successfully. A1, B and A2 each completed 36 input rows.
All three comparisons produced 27/27 finite output pair byte
identity, 36/36 equal states and 36/36 equal covariance
Cholesky statuses. The full result is stored as
\`comparison.json\` plus \`a1.json\`, \`b.json\`, \`a2.json\`.

**Interpretation:** This one-job matched host eliminates the host
difference *within that job*. It demonstrates that neither
the higher Python version nor the higher Torch wheel **forces**
different bits on this machine and workload. It does not prove
cross-host reproducibility or identify the reason other
GitHub runner jobs disagreed.

## Independent witness and resource-parity audit

The 100-digit mpmath posterior computation is arithmetically
independent of Sionna's LMMSE solve path but shares the same
synthetic input fixtures. The raw frozen source and transported
covariances, input hashes and solver outputs are independent of
any editorial aggregate. The A2 return phase is a **matched
same-version control**.

Both virtualenvs run on the same hosted VM, are single-threaded and
load the same fixture bytes. This is a numerical **comparison**
rather than a runtime performance benchmark; we do not infer
latency, power efficiency or physical receiver quality.

## Round-trip audit and corrections

Requirement: Distinguish version-specific numerical effects from
unmatched hardware effects. It became an exact-byte fixture,
then a same-runner A1→B→A2 implementation, then raw output/error
measurement. The result supports **bounded cross-runtime equality
on the matched host**, not a global portability guarantee. Earlier
opposing observations remain in the record:

- [First separate-host replay](https://github.com/jesusvilela/sionna/actions/runs/44048790316):
  9/9 finite fixture/mode output pairs byte-identical between versions.
- [Second separate-host replay](https://github.com/jesusvilela/sionna/actions/runs/44049760075):
  0/9 finite fixture/mode output pairs byte-identical between versions.
- [Matched one-host A-B-A](https://github.com/jesusvilela/sionna/actions/runs/44054491046):
  27/27 finite repeated rows byte-identical in all A1/B, B/A2 and
  A1/A2 comparisons.

The difference between **nine fixture/mode combinations** and
**27 repeated rows** is essential: the latter are not independent
samples. The six selected fixtures are also not statistically
representative of practical radio channels.

## Strongest surviving result / most damaging obstruction

**Strongest:** A fixed synthetic fixture and one matched GitHub VM
yield byte-identical finite LMMSE outputs across the two tested
runtime stacks and after reverting to the first stack, with
identical independent oracle outputs and explicit nonfinite cases.

**Obstruction:** Cross-host observations disagree in the finite
output bits under identical represented inputs. The CPU model,
virtualization, runner microarchitecture, ISA, and vendor-library
dispatch were not matched in those earlier separate-host runs.
We cannot assign causal responsibility to any particular factor
from one additional matched-host experiment.

## Next bounded experiment and stop rule

**Object:** only if an upstream maintainer or physical receiver trace
motivates further work, repeat the same frozen fixture on a
second explicitly identified physical CPU type with both Torch
builds under the **same Python version**.

**Prediction:** the matched-host A-B-A within-job comparison will
remain repeatable on the new hardware, while cross-hardware
differences may occur in the original/transport product.

**Primary metric:** bytewise original and transported output SHA
identity plus raw complex-estimate/variance deltas; Cholesky status
and fixture SHA must be recorded separately.

**Controls:** same fixture hash, fixed Python, same documented
Torch/BLAS packaging, A-B-A same-version reversal, three repeats,
and the moderate SPD controls.

**Kill condition:** if A1 and A2 differ on the same host,
a simple "runtime-version causal" attribution is invalid.
If any fixture digest differs, no numerical comparison is valid.

**Stop condition:** absent realistic measured receiver inputs,
supported-device reproduction of a failure, or upstream request,
freeze this research rather than modifying production Sionna.

**Promotion rule:** do not open a numerical-fix upstream PR without
an independently reproduced supported-device defect and a solution
with matched ordinary-case cost.

## Evidence

- [A-B-A implementation](experiments/compare_lmmse_matched_host.py)
- [Workflow with pinned environments](../.github/workflows/research-lmmse-matched-host-aba.yml)
- [Committed fixture](fixtures/lmmse_tensor_replay_v1.json)
- [Prior corrected frozen replay audit](LMMSE_FROZEN_REPLAY_AUDIT_2026-11-17.md)
- [Exact CI run and 30-day raw evidence artifact](https://github.com/jesusvilela/sionna/actions/runs/44054491046)

**No upstream change.** [NVlabs/sionna PR #1208](https://github.com/NVlabs/sionna/pull/1208)
and [NVlabs/sionna-rt PR #82](https://github.com/NVlabs/sionna-rt/pull/82)
remain separate, awaiting maintainers.
