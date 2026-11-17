# Fixed-Python Sionna LMMSE A-B-A reproducibility audit

**17 November 2026. Research-only, not a production proposal.**

Exact [Actions run 44064419391](https://github.com/jesusvilela/sionna/actions/runs/44064419391); tested source commit `a3b72845fb6e8fe3d804b2e74873d1381208d233`; frozen fixture SHA-256 `f6a943a80f49286de247f8327e620f2155086fda88210acfcef9863951b76995`.

## Verdict

Holding **Python 3.11.18**, the GitHub-hosted CPU and six frozen complex input fixtures constant, PyTorch **2.9.1** and **2.10.0** produced identical finite LMMSE output bytes in an A1 → B → A2 experiment. All finite output differences were zero, all failure classifications and Cholesky statuses matched, and both moderately conditioned controls passed. However, the stress cases still included nonfinite and inaccurate outcomes. Thus **bitwise agreement does not imply accuracy**, and this single-host result does not establish global CPU/GPU portability or a physical 6G benefit.

## Claim and evidence ledger

| Claim | Before | Artifact | Result | After |
|---|---|---|---|---|
| Coherent invertible frame transport preserves exact LMMSE statistics | P conditional | `S'=WSW†` posterior identity | Survives under SPD and normalization assumptions | P |
| Fixed-Python, fixed-host Torch releases can agree byte-for-byte on selected data | H | A1-B-A2 run 44064419391 | 27/27 finite repeated rows equal in every comparison | M |
| Tested moderately conditioned outputs meet oracle tolerance | M | 100-digit mpmath represented-input posterior | 12/12 safe repeated rows passed per stage | M |
| A Torch version change necessarily changes output bits | H | Same-host A1 vs B | False as a universal assertion | R |
| Severe conditioning is always numerically safe | H | Extreme frozen fixtures | 9 nonfinite repeated rows plus 9 finite extreme rows out of tolerance per stage | R |
| CPU ISA uniquely caused earlier cross-host differences | H | Different runner logs | Not isolated | H |
| This proves an RF/6G signal quality improvement | H | No receiver trace | Not supported | H |

## Test object and independent witness

The experiment fixes source and transported observations, channels, covariance matrices, and receiver-frame operators as serialized `complex64` / `complex128` little-endian bytes. Every replay verifies SHA-256, dtype and shape. Six purposively selected fixtures × two whitening modes × three repeats = **36 rows per stage**, of which **27 are finite** (nine distinct fixture/mode combinations). Three extreme whitening configurations generate nine repeated nonfinite rows. These are **not independent RF observations**.

A separate 100-digit mpmath posterior oracle evaluates the *actual represented inputs*. It is computationally different from Sionna's LMMSE path, but shares the same synthetic-data provenance and does not independently validate physical performance.

## Matched-control results

| Stage | Python | PyTorch |
|---|---|---|
| A1 | 3.11.18 | 2.9.1+cpu |
| B | 3.11.18 | 2.10.0+cpu |
| A2 | 3.11.18 | 2.9.1+cpu |

All executed in **one GitHub runner job** on an **AMD EPYC 7763**, PyTorch-reported **AVX2**, one CPU thread. The non-Torch numerical packages were pinned identically. Both wheels reported Intel MKL 2024.2 and oneDNN 3.7.1; distinct Torch binaries and native kernel dispatch remain a confound.

| Matched comparison | Equal finite output bytes | Equal states | Equal Cholesky statuses | Max finite numeric difference |
|---|---:|---:|---:|---:|
| A1 vs B | 27/27 | 36/36 | 36/36 | 0 |
| B vs A2 | 27/27 | 36/36 | 36/36 | 0 |
| A1 vs A2 | 27/27 | 36/36 | 36/36 | 0 |

The A1 → A2 return-to-original comparison is the crucial same-version control. No unsafe cases were discarded to obtain these counts. Note that nine finite extreme repeated rows **failed the high-precision accuracy tolerance even while both versions returned identical bits**.

## Representation attack and load-bearing obstruction

A diagnostic recomputed `W @ S @ W.mH`, without replacing any frozen solver input:

- Current **AMD/AVX2** job: both Torch versions recomputed all six covariance fixtures with bytes matching the originally stored `sp` arrays.
- Earlier [Intel Xeon 8573C / AVX-512 same-host experiment](https://github.com/jesusvilela/sionna/actions/runs/44054491046): both versions agreed with one another but neither matched the originally stored `sp` bytes in any of six fixtures.
- Earlier **separate-host** runs yielded conflicting finite output bytes under identical frozen inputs.

The changing host is a plausible contributor to these differences. But **AMD vs Intel, ISA, virtual machine, native library dispatch, and host configuration are not independently controlled**. It would be an invalid causal promotion to label any one of them the demonstrated cause.

## Specification -> implementation -> test -> interpretation

**Specification:** Freeze input bytes, physical runner and Python interpreter, change only the PyTorch wheel environment in an A1-B-A2 order, preserve raw mismatches and failures.

**Implementation:** Two isolated Python 3.11 venvs, same library pins except Torch, call Sionna's public `lmmse_equalizer` and the independent represented-input mpmath oracle. The comparison script rejects differing fixture hashes or unexpected Python/Torch versions.

**Test:** All 36 states per stage, bytewise output hashes, separate symbol-estimate and effective-noise errors, Cholesky statuses and recomputed transport hashes. No latency benchmark and no RF evaluation.

**Interpretation:** In this selected experiment the two Torch releases happen to agree on all finite outputs; severe-condition inaccuracy and invalid transformed covariances remain. Repeatability cannot be promoted into numerical correctness or domain benefit.

## Round-trip correction

The prior matched-host A-B-A changed Python **3.11 → 3.13** alongside Torch **2.9.1 → 2.10.0**. This experiment removes the Python-version difference while preserving the same frozen object. The output agrees, therefore the Python-version change is **not needed** to obtain cross-version equality on the tested hosts. It does not show that Python 3.13 could never affect a different workload.

## Bounded next experiment / stop rule

**Object:** Only if a concrete upstream review or supported receiver trace motivates further work, execute the frozen fixture on one CPU with explicit ISA/vendor-library dispatch controls.

**Prediction:** Recomputed covariance product bytes may change when backend dispatch changes, while frozen-input solver outputs need not.

**Primary metric:** SHA-256 of recomputed covariance, raw LMMSE complex estimates and effective-noise variances, Cholesky status and mpmath accuracy, each recorded separately.

**Controls:** Same fixture SHA, same Python and CPU, return-to-original A1/A2, identical library pins, three repeated calls.

**Kill:** A1 and A2 mismatch or any fixture digest differs; then causal attribution to the intended changed factor fails.

**Stop:** Do not run more synthetic expansions without upstream maintainer feedback, a supported-device defect or real receiver data. Do not open another upstream PR.

**Promotion:** A numerical production change would need a realistic, replicated receiver problem and a solution with acceptable matched ordinary-case latency.

## Evidence tags

**P** = exact receiver-frame LMMSE identity under SPD/invertibility assumptions. **A** = hypotheses and precision tolerances. **M** = this bounded CPU replay, including failures. **H** = unresolved host/ISA causality and physical 6G significance. **S** = receiver moving frame as coordinate representation, not new propagation physics. **R** = universal bitwise-portability and universal numerical-robustness claims.

Evidence: [workflow](../.github/workflows/research-lmmse-python311-only-aba.yml), [comparison code](experiments/compare_lmmse_matched_host.py), [frozen fixture](fixtures/lmmse_tensor_replay_v1.json), [exact run and raw artifacts](https://github.com/jesusvilela/sionna/actions/runs/44064419391), [previous matched-host audit](LMMSE_MATCHED_HOST_ABA_AUDIT_2026-11-17.md). Upstream [PHY #1208](https://github.com/NVlabs/sionna/pull/1208) and [RT #82](https://github.com/NVlabs/sionna-rt/pull/82) remain untouched.
