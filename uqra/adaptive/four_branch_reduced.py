"""Deterministic reduced FourBranch software benchmark.

This is a newly generated software fixture, not a reconstruction of the
unavailable historical FourBranch candidate or test data.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json

import numpy as np

from .controller import AdaptiveSparsePCE
from .profiles import publication_profile
from .state import array_hash


BENCHMARK_NAME = "four_branch_reduced_v1"
SCENARIO = "reduced"
CANDIDATE_SEED = 26080501
TEST_SEED = 26080502
REFERENCE_SEED = 26080503
CV_SEED = 26080504
SIZES = {"candidate": 192, "test": 256, "reference": 4096}
INPUT_HASHES = {
    "candidate": "01ca05dc7f0312570cd13e6514b869683e1f7247b42c4dae72b5a1c05281538a",
    "test": "2ea6903f46a4e79f8d3844221a6dae21c16a28328b42150046eec3eed8058b42",
    "reference": "96d87567367d58e82dff38f84e05a01a608a1a6b7ba5a3491b012e67d2f5f5db",
}


def four_branch_limit_state(xi):
    """Return g=min(g1,...,g4); failure is explicitly g <= 0."""
    xi = np.asarray(xi, dtype=float).reshape(2, -1)
    x1, x2 = xi
    root2 = np.sqrt(2.0)
    branches = np.vstack((
        3.0 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / root2,
        3.0 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / root2,
        x1 - x2 + 7.0 / root2,
        x2 - x1 + 7.0 / root2,
    ))
    return np.min(branches, axis=0)


def frozen_inputs():
    """Return independent, read-only candidate, test, and reference arrays."""
    arrays = {
        "candidate": np.random.default_rng(CANDIDATE_SEED).normal(size=(2, SIZES["candidate"])),
        "test": np.random.default_rng(TEST_SEED).normal(size=(2, SIZES["test"])),
        "reference": np.random.default_rng(REFERENCE_SEED).normal(size=(2, SIZES["reference"])),
    }
    for array in arrays.values():
        array.setflags(write=False)
    actual = {name: array_hash(array) for name, array in arrays.items()}
    if actual != INPUT_HASHES:
        raise RuntimeError(f"FourBranch reduced frozen input hash mismatch: {actual}")
    return arrays


def _vandermonde(order, xi):
    from uqra.polynomial.hermite import Hermite
    return Hermite(d=2, deg=order, hem_type="probabilists").vandermonde(xi)


def _identity_payload(arrays):
    seeds = {"candidate": CANDIDATE_SEED, "test": TEST_SEED, "reference": REFERENCE_SEED}
    return {
        name: {
            "role": name,
            "rng": "numpy.random.Generator(PCG64)",
            "seed": seeds[name],
            "shape": list(array.shape),
            "sha256": array_hash(array),
        }
        for name, array in arrays.items()
    }


def run_scenario():
    arrays = frozen_inputs()
    candidate, test, reference = (arrays[name] for name in ("candidate", "test", "reference"))
    test_g = four_branch_limit_state(test)
    boundary_ids = np.argsort(np.abs(test_g), kind="stable")[:8]
    profile = publication_profile(
        cv_folds=4, cv_seed=CV_SEED, qoi_tolerance=1e-3,
        outer_stable_checks=1, inner_qoi_tolerance=None,
        max_inner_iterations=3, minimum_doi_size=8, overfit_rebuild=False,
    )
    runner = AdaptiveSparsePCE(
        candidate, _vandermonde, four_branch_limit_state, profile,
        order_min=2, order_max=4, test_xi=test,
        qoi=lambda values: np.mean(np.asarray(values) <= 0.0),
        doi_centers=test[:, boundary_ids], doi_radius=0.75, criterion="S",
        random_state=CANDIDATE_SEED,
    )
    result = runner.run()
    result.state.assert_invariants()
    stages = Counter(row.stage for row in result.trace)
    return {
        "scenario": SCENARIO,
        "status": result.status,
        "stop_reason": result.stop_reason,
        "candidate_hash": array_hash(candidate),
        "test_hash": array_hash(test),
        "reference_hash": array_hash(reference),
        "reference_failure_probability": float(np.mean(four_branch_limit_state(reference) <= 0.0)),
        "doi_center_test_ids": boundary_ids.tolist(),
        "trace_hash": result.trace_hash(),
        "trace_rows": len(result.trace),
        "stage_counts": dict(sorted(stages.items())),
        "model_call_count": result.state.model_call_count,
        "evaluated_global_ids": list(result.state.evaluated_global_ids),
        "evaluated_coordinate_hashes": sorted(result.state.evaluated_coordinate_hashes),
        "trace": [asdict(row) for row in result.trace],
    }


def run_suite(scenarios=None):
    scenarios = list(scenarios or (SCENARIO,))
    if scenarios != [SCENARIO]:
        raise ValueError(f"{BENCHMARK_NAME} supports only the {SCENARIO!r} scenario")
    arrays = frozen_inputs()
    manifest = {
        "schema_version": 1,
        "benchmark": BENCHMARK_NAME,
        "input": {
            "purpose": "software_benchmark",
            "scale": "reduced",
            "historical_replay": False,
            "failure_definition": "min(g1,g2,g3,g4) <= 0",
            "cv_seed": CV_SEED,
            "datasets": _identity_payload(arrays),
        },
        "scenarios": {SCENARIO: run_scenario()},
    }
    manifest["stable_manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return manifest
