"""Compare fixed-input LMMSE replays in an A-B-A sequence on one runner.

A1: Python 3.11 / PyTorch 2.9.1.
B:  Python 3.11 / PyTorch 2.10.0 (same Python minor version).
A2: Python 3.11 / PyTorch 2.9.1 again.

No solver outputs are repaired, rounded, clipped or discarded. The complete
per-config comparisons are retained, including nonfinite states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


STAGES = ("a1", "b", "a2")
TENSOR_ROLES = ("original_output", "transported_output")
EXPECTED_VERSIONS = ("2.9.1", "2.10.0", "2.9.1")
EXPECTED_PYTHON = "3.11"


def _read(path: Path, expected_version: str) -> dict:
    raw = path.read_bytes()
    data = json.loads(raw)
    if not data["protocol"]["torch"].startswith(expected_version):
        raise ValueError("Unexpected Torch version in " + str(path))
    if not data["protocol"]["python"].startswith(EXPECTED_PYTHON):
        raise ValueError("Unexpected Python version in " + str(path))
    if data["protocol"]["repeats"] != 3:
        raise ValueError("Expected three repeated frozen-input replays")
    if len(data["rows"]) != 36:
        raise ValueError("Expected exactly 6 cases x 2 modes x 3 repeats")
    data["_source_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def _key(row: dict) -> tuple:
    return (row["id"], row["whiten_interference"], row["repeat"])


def _indexed(data: dict) -> dict:
    result = {}
    for row in data["rows"]:
        key = _key(row)
        if key in result:
            raise ValueError("Duplicate replay row: " + repr(key))
        result[key] = row
    return result


def _output_difference(x: list, y: list) -> dict:
    """Report raw absolute errors for distinct typed observables."""
    x_estimate, x_eff_noise = x
    y_estimate, y_eff_noise = y
    if len(x_estimate) != len(y_estimate):
        raise ValueError("Differing estimate stream count")
    if len(x_eff_noise) != len(y_eff_noise):
        raise ValueError("Differing effective-noise stream count")
    diffs = [
        abs(complex(*a) - complex(*b))
        for a, b in zip(x_estimate, y_estimate)
    ]
    noise_diffs = [
        abs(a - b) for a, b in zip(x_eff_noise, y_eff_noise)
    ]
    return {
        "complex_estimate_absolute_errors": diffs,
        "effective_noise_variance_absolute_errors": noise_diffs,
        "max_estimate_abs_error": max(diffs),
        "max_effective_noise_abs_error": max(noise_diffs),
    }


def _oracle_delta(a: dict, b: dict) -> dict:
    return {
        "original": _output_difference(
            a["original_oracle"], b["original_oracle"]
        ),
        "transported": _output_difference(
            a["transported_oracle"], b["transported_oracle"]
        ),
    }


def _pair_compare(a: dict, b: dict, label: str) -> dict:
    rows_a = _indexed(a)
    rows_b = _indexed(b)
    if rows_a.keys() != rows_b.keys():
        raise ValueError("Input row keys differ")

    cases = []
    for key in sorted(rows_a, key=lambda x: (x[0], x[1], x[2])):
        x, y = rows_a[key], rows_b[key]
        match = x["state"] == y["state"]
        comparison = {
            "id": key[0],
            "whiten_interference": key[1],
            "repeat": key[2],
            "state_a": x["state"],
            "state_b": y["state"],
            "state_equal": match,
            "base_cholesky_info_equal": (
                x["base_diagnostics"]["cholesky_info"]
                == y["base_diagnostics"]["cholesky_info"]
            ),
            "transported_cholesky_info_equal": (
                x["transported_diagnostics"]["cholesky_info"]
                == y["transported_diagnostics"]["cholesky_info"]
            ),
            "frozen_transport_a_bytes_equal": (
                x["recomputed_transport_bytes_equal"]
            ),
            "frozen_transport_b_bytes_equal": (
                y["recomputed_transport_bytes_equal"]
            ),
            "frozen_transport_a_sha256": (
                x["recomputed_transport_sha256"]
            ),
            "frozen_transport_b_sha256": (
                y["recomputed_transport_sha256"]
            ),
            "original_oracle_delta": _oracle_delta(x, y)["original"],
            "transported_oracle_delta": _oracle_delta(x, y)["transported"],
        }
        if x["state"] == "finite" and y["state"] == "finite":
            original_equal = (
                x["original_result_sha256"]
                == y["original_result_sha256"]
            )
            transported_equal = (
                x["transformed_result_sha256"]
                == y["transformed_result_sha256"]
            )
            comparison.update({
                "original_output_bytes_equal": original_equal,
                "transported_output_bytes_equal": transported_equal,
                "both_output_bytes_equal": (
                    original_equal and transported_equal
                ),
                "original_output_delta": _output_difference(
                    x["original_output"], y["original_output"]
                ),
                "transported_output_delta": _output_difference(
                    x["transported_output"], y["transported_output"]
                ),
                "original_output_sha256_a": x["original_result_sha256"],
                "original_output_sha256_b": y["original_result_sha256"],
                "transported_output_sha256_a": (
                    x["transformed_result_sha256"]
                ),
                "transported_output_sha256_b": (
                    y["transformed_result_sha256"]
                ),
            })
        cases.append(comparison)

    finite = [x for x in cases if "both_output_bytes_equal" in x]
    out = {
        "comparison": label,
        "total": len(cases),
        "same_state": sum(c["state_equal"] for c in cases),
        "same_base_cholesky": sum(
            c["base_cholesky_info_equal"] for c in cases
        ),
        "same_transported_cholesky": sum(
            c["transported_cholesky_info_equal"] for c in cases
        ),
        "both_finite": len(finite),
        "both_finite_byte_identical": sum(
            c["both_output_bytes_equal"] for c in finite
        ),
        "max_original_estimate_abs_delta": max(
            (c["original_output_delta"]["max_estimate_abs_error"]
             for c in finite), default=None
        ),
        "max_transported_estimate_abs_delta": max(
            (c["transported_output_delta"]["max_estimate_abs_error"]
             for c in finite), default=None
        ),
        "max_original_effective_noise_abs_delta": max(
            (c["original_output_delta"]["max_effective_noise_abs_error"]
             for c in finite), default=None
        ),
        "max_transported_effective_noise_abs_delta": max(
            (c["transported_output_delta"]["max_effective_noise_abs_error"]
             for c in finite), default=None
        ),
        "raw_comparisons": cases,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in STAGES:
        parser.add_argument("--" + stage, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()

    data = {
        stage: _read(getattr(opts, stage), version)
        for stage, version in zip(STAGES, EXPECTED_VERSIONS)
    }
    fixture_hashes = {
        stage: value["summary"]["fixture_sha256"]
        for stage, value in data.items()
    }
    tensor_hashes = {
        stage: value["summary"]["fixture_input_sha256"]
        for stage, value in data.items()
    }
    if len(set(fixture_hashes.values())) != 1:
        raise AssertionError("FATAL: fixture file digest changed")
    fingerprints = [
        json.dumps(v, sort_keys=True) for v in tensor_hashes.values()
    ]
    if len(set(fingerprints)) != 1:
        raise AssertionError("FATAL: raw tensor input fingerprints changed")

    for stage in STAGES:
        if data[stage]["summary"]["safe_failures"] != 0:
            raise AssertionError(stage + " moderate control failed")

    comparisons = {
        "a1_vs_b": _pair_compare(data["a1"], data["b"], "A1 vs B"),
        "b_vs_a2": _pair_compare(data["b"], data["a2"], "B vs A2"),
        "a1_vs_a2": _pair_compare(data["a1"], data["a2"], "A1 vs A2"),
    }
    # Replaying A after B in the same VM is a matched same-version control.
    # Do NOT assume bits match; this is the explicit causal-inference gate.
    same_version_stable = (
        comparisons["a1_vs_a2"]["both_finite_byte_identical"]
        == comparisons["a1_vs_a2"]["both_finite"]
        and comparisons["a1_vs_a2"]["same_state"]
        == comparisons["a1_vs_a2"]["total"]
    )
    summary = {
        "input_fixture_sha256": next(iter(fixture_hashes.values())),
        "source_json_sha256": {
            stage: value["_source_sha256"] for stage, value in data.items()
        },
        "runtime_versions": {
            stage: value["protocol"]["torch"] for stage, value in data.items()
        },
        "python_versions": {
            stage: value["protocol"]["python"] for stage, value in data.items()
        },
        "same_input_tensor_bytes": True,
        "a1_vs_a2_same_version_bitwise_stable": same_version_stable,
        "cross_version_causal_interpretation_allowed": (
            same_version_stable
        ),
        "comparison_summary": {
            key: {
                k: v for k, v in comparison.items()
                if k != "raw_comparisons"
            }
            for key, comparison in comparisons.items()
        },
        "limitation": (
            "Matched one GitHub VM and Python 3.11 controls the host "
            "and interpreter minor version. PyTorch wheels may bundle "
            "different numerical libraries or dispatch implementations. "
            "No single underlying library can be causally isolated here."
        ),
    }
    result = {"summary": summary, "comparisons": comparisons}
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("MATCHED_HOST_ABA_SUMMARY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
