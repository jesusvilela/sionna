# Adaptive covariance whitening audit — 15 November 2026

## Verdict

A scalar rescaling of the covariance is an exact representation change: it can move a positive-definite covariance out of the subnormal/top-exponent region without changing the exact whitened observation or channel, provided the same square-root scale is applied to both. The research implementation passes its 15 numerical, mixed-batch, and graph-capture tests on both supported CPU runtime combinations. However, every branchless adaptive implementation measured here is materially slower than Sionna's baseline on ordinary-range inputs, and attempts to create a zero-overhead tensor conditional have not produced a viable batched fast path: eager torch.cond is slower, while the batched Inductor benchmark fails in generated C++ compilation. **No production change is justified.** Keep this as a research-only negative/partial result and freeze it unless a deployment actually exhibits this scale pathology.

## Frozen claim and intended observable

**Claim under test:** For a Hermitian positive-definite covariance \(S\), there may be an adaptive scalar representation \(S_u=S/\alpha\) and corresponding data scaling \(y_u=y/\sqrt{\alpha}\), \(H_u=H/\sqrt{\alpha}\) that stabilizes whitening at extreme floating-point exponents while preserving the output of Sionna's existing whitening path in exact arithmetic.

**Observable:** the whitened observation \(L^{-1}y\), whitened channel \(L^{-1}H\), their Gram \(H^\dagger S^{-1}H\), and candidate-codeword Mahalanobis distances. The implementation under study is research-only and does not modify Sionna production code.

## Exact mathematical identity

Let \(S\succ0\) and \(S=\alpha S_u\) for \(\alpha>0\). If \(S_u=L_uL_u^\dagger\), then \(S=(\sqrt{\alpha}L_u)(\sqrt{\alpha}L_u)^\dagger\). Thus

\[
(\sqrt{\alpha}L_u)^{-1}y=L_u^{-1}(y/\sqrt{\alpha}),
\qquad
(\sqrt{\alpha}L_u)^{-1}H=L_u^{-1}(H/\sqrt{\alpha}).
\]

Consequently, in exact arithmetic the whitened outputs, Gram and all residual distances are unchanged. This is a scalar exponent-range transformation only:

\[
\kappa_2(S/\alpha)=\kappa_2(S).
\]

It does **not** improve the spectral condition number. It cannot repair a genuinely ill-conditioned covariance, a rank-deficient matrix, or inaccurate input data.

## Implementation candidates

1. **Power-of-two scaling:** use the maximum component exponent, rescale \(S\) into a unit exponent interval, and compensate \(y,H\) by the corresponding half-exponent. A direct torch.ldexp on subnormal components produced nonfinite whitening outputs in the tested CPU environment. Decomposing each real/imaginary component with frexp before ldexp restored finite results and passed the high-precision checks.
2. **Componentwise real/imaginary division:** divide the real and imaginary covariance components separately by a real positive scale, and divide \(y,H\) by its square root. This avoids complex division and also passes the selected extreme-scale oracles.
3. **Conditional fast path:** a tensor predicate was explored to keep the production path for normal inputs. Passing complex tensors directly to torch.cond hit a real/imaginary view aliasing restriction. Passing view_as_real operands avoids that aliasing and passes small-shape graph capture/Inductor correctness, but eager execution is slower and batched Inductor compilation failed in generated C++.

All code remains on branch \`research/adaptive-power2-whitening-2026-11-15\`:
- \`research/experiments/adaptive_whitening.py\`
- \`research/tests/test_adaptive_whitening.py\`
- \`.github/workflows/research-adaptive-whitening.yml\`

## Verification

Latest passing numerical/benchmark run: [GitHub Actions 43755135679](https://github.com/jesusvilela/sionna/actions/runs/43755135679), exact head \`c768437d2a4d2eded0f400ed765b47b15bb659c1\`.

| Runtime | Research tests | Result |
|---|---:|---|
| Python 3.11 / PyTorch 2.9.1 CPU | 15 | Passed |
| Python 3.13 / PyTorch 2.10.0 CPU | 15 | Passed |

The tests cover:
- exact no-op equivalence to production whitening on normal-range covariance;
- complex128 subnormal covariance against a 100-digit mpmath oracle on the actual represented inputs, including Gram and candidate distances;
- complex128 top-exponent covariance against the same oracle;
- complex64 subnormal covariance against an independent complex128 solve;
- mixed batches with normal and subnormal covariances and broadcasted batch shapes;
- fullgraph capture for the tensor-only power-of-two and componentwise-division candidates;
- fullgraph and small-shape Inductor execution for the real-view torch.cond candidate.

These are CPU-only synthetic matrix tests, not receiver measurements. Python 3.13 reported upstream PyTorch deprecation warnings; they were nonfatal.

## Matched ordinary-scale CPU benchmark

Single-thread CPU, complex128, eager calls, warmup 10, five timing samples of 40 calls each. All paths return only whitened y/H (no return_s allocation). Matrices were synthetic SPD samples \(S=CC^\dagger+0.5I\), with random y/H. Times are medians in milliseconds per call.

| Matrix / batch | Production baseline | Adaptive division | Power-of-two | Conditional view, eager |
|---|---:|---:|---:|---:|
| 2x2 / 128 | 0.049–0.080 | 0.187–0.317 | 0.277–0.463 | 0.437–0.642 |
| 2x2 / 1024 | 0.233–0.327 | 0.434–0.639 | 0.658–1.046 | 0.669–0.964 |
| 8x8 / 128 | 0.164–0.221 | 0.340–0.514 | 0.581–0.939 | 0.605–0.893 |
| 8x8 / 1024 | 1.135–1.478 | 2.159–3.126 | 4.338–6.635 | 1.676–2.296 |

Ranges show the two tested runtime combinations, not confidence intervals. On these ordinary-range cases, adaptive division costs roughly **1.5–4.0x** the baseline, power-of-two scaling roughly **2.8–5.8x**, and eager conditional-view execution roughly **1.5–8.9x**. These are local CPU timings only. They do not predict GPU or end-to-end detector latency.

A batched Inductor benchmark for the conditional-view path was attempted separately. It failed in generated C++ with a vectorized integer addition type mismatch (\`VectorizedN<int, 2>\` versus \`Vectorized<int>\`) in both tested PyTorch environments. This is recorded as an environment/compiler obstruction, not a mathematical counterexample. The small-shape Inductor correctness test itself passed.

## Claim and evidence ledger

| Claim | Tag | Evidence / limitation |
|---|---|---|
| Positive scalar covariance scaling preserves exact whitened outputs | **P** | Algebra above |
| Selected power-of-two and componentwise-division candidates recover finite accurate outputs at tested extremes | **M** | CPU tests; mpmath oracle for complex128 actual inputs, complex128 solve for complex64 |
| Normal-range adaptive paths are bitwise equal to the baseline outputs | **M** | Explicit exact-equality test |
| Branchless adaptive paths are production improvements | **R** | Retired: ordinary-scale CPU overhead is material |
| torch.cond is a usable zero-overhead eager fast path | **R** | Retired: alias issue for complex operands; real-view eager path slower |
| Batched Inductor conditional fast path is validated | **R** | Not established; generated C++ compilation failed |
| Physical receiver sensitivity, BER/SER or Munich calibration improves | **H/R** | No physical dataset or RF experiment |

## Specification–implementation–test audit

- The algebraic requirement is correct and independent of the implementation.
- The tests use an independent high-precision oracle for the complex128 extreme cases and an independent complex128 solve for the complex64 case.
- The mixed-batch test checks that each covariance receives its own exponent and that the exponent broadcasts across additional y/H batch dimensions.
- The normal-range no-op requirement is tested exactly for the power-of-two and division candidates.
- The implementation has not changed Sionna's production path; all code is isolated in a research branch.
- The most damaging obstruction is resource cost: the adaptive paths do not satisfy the original goal of avoiding ordinary-input overhead. The conditional experiment does not currently repair this.

## Decision and stop rule

**Decision:** do not open a production PR and do not merge this branch into the receiver-frame regression PR. Preserve the code and negative results as research artifacts.

**One bounded next experiment, only if an actual deployment needs it:** investigate a fused C++/Triton operation that performs exponent classification and the rare scaling path without materializing multiple real/imaginary intermediates. Compare against the exact production path on normal and extreme batches, including GPU graph capture and end-to-end ML detector outputs.

- **Prediction:** a fused implementation can keep normal-range latency within 5% of baseline while retaining the measured extreme-scale accuracy.
- **Primary metric:** paired normal-range median latency and extreme-scale error against the actual-input high-precision oracle.
- **Kill condition:** more than 5% normal-range overhead, loss of oracle accuracy, or failure of graph capture.
- **Stop condition:** no concrete deployment input demonstrates the scale pathology; do not spend further engineering effort on an exotic synthetic edge case.
- **Promotion rule:** only an independently reproduced defect on supported hardware and a production-relevant workload can justify upstream consideration.

The original research claim is therefore narrowed: scalar adaptive scaling can repair selected floating-point exponent failures, but the current tensor-level implementations are not production-worthy because their ordinary-case cost is too high.


## Upstream data-path relevance check

A source inspection of the current Sionna PHY tree found these relevant paths:

- MaximumLikelihoodDetector.call invokes whiten_channel(y, h, s, return_s=False).
- lmmse_equalizer also has a whitening path for its supplied covariance.
- OFDM compute_interference_covariance_matrix forms \(S=\operatorname{diag}(no)+H_{\rm undesired}H_{\rm undesired}^\dagger\) when those inputs are supplied.
- The AWGN block accepts a caller-supplied variance no and scales generated noise by \(\sqrt{no}\). The inspected source does not impose a positive lower exponent bound on no.

Thus subnormal S is representable as an extreme caller-supplied case when interference is absent or negligible, but no inspected example or test demonstrates that supported operating workloads routinely approach the subnormal range. The documentation example uses no=0.1; this is illustrative, not a distributional survey. There is no authenticated Munich receiver covariance dataset in this experiment.

This further weakens the case for paying a universal runtime cost to handle a synthetic exponent edge case. A real deployment trace showing subnormal or top-bin covariance inputs is the prerequisite for the proposed fused follow-up.
