"""Deterministic end-to-end benchmark and manifest generator for adaptive PCE."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
import sklearn

from .controller import AdaptiveSparsePCE
from .profiles import publication_profile
from .state import array_hash


BENCHMARK_NAME = "phase8_two_dimensional_hermite"
BENCHMARK_SEED = 424242
CV_SEED = 8080
REPRODUCE_COMMAND = (
    "python -m uqra.adaptive.benchmark --scenario all "
    "--output artifacts/adaptive_phase8_manifest.json"
)


def _hermite_vandermonde(order, xi):
    # Import here so the module remains usable without coupling controller code to uqra.__init__.
    from uqra.polynomial.hermite import Hermite
    return Hermite(d=2, deg=order, hem_type="probabilists").vandermonde(xi)


def _benchmark_model(xi):
    return 1.0 + 0.8 * xi[0] + 0.35 * (xi[1] ** 2 - 1.0) + 0.1 * xi[0] * xi[1]


def frozen_benchmark_inputs():
    rng = np.random.default_rng(BENCHMARK_SEED)
    candidate = rng.normal(size=(2, 96))
    test = rng.normal(size=(2, 128))
    candidate.setflags(write=False); test.setflags(write=False)
    return candidate, test


class _ScheduledCVRunner(AdaptiveSparsePCE):
    """Harness-only deterministic overfitting trigger using real fits and fixed CV labels."""

    cv_schedule = {1: 1.0, 2: 2.0, 3: 3.0}

    def _fit(self, order, X):
        fit = super()._fit(order, X)
        fit.cv_error = self.cv_schedule[int(order)]
        self.state.cv_path = list(fit.cv_path)
        return fit


def _git_metadata():
    root = Path(__file__).resolve().parents[2]
    def run(*args):
        process = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args],
                                 capture_output=True, text=True, check=False)
        return process.stdout.strip() if process.returncode == 0 else None
    status = run("status", "--porcelain")
    return {
        "repository_root": str(root),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "worktree_dirty": None if status is None else bool(status),
    }


def _profile_for(scenario):
    common = dict(cv_folds=4, cv_seed=CV_SEED, inner_qoi_tolerance=None,
                  max_inner_iterations=3, minimum_doi_size=4)
    if scenario == "converged":
        return publication_profile(qoi_tolerance=1e9, outer_stable_checks=1,
                                   overfit_rebuild=False, **common)
    if scenario == "max_order":
        return publication_profile(qoi_tolerance=1e-12, outer_stable_checks=1,
                                   overfit_rebuild=False, **common)
    if scenario == "overfit_fallback":
        return publication_profile(qoi_tolerance=1e-12, outer_stable_checks=99,
                                   overfit_rebuild=True, **common)
    if scenario == "runtime_failure":
        return publication_profile(qoi_tolerance=1e-12, outer_stable_checks=1,
                                   overfit_rebuild=False, **common)
    raise ValueError(f"unknown benchmark scenario: {scenario}")


def _validate_identity(result):
    result.state.assert_invariants()
    seen = set()
    additions_are_new = True
    trace_is_cumulative = True
    candidate_hash_stable = True
    for record in result.trace:
        added = set(record.added_global_ids)
        additions_are_new &= not bool(added.intersection(seen))
        evaluated = set(record.evaluated_global_ids)
        trace_is_cumulative &= seen.issubset(evaluated)
        candidate_hash_stable &= record.candidate_hash == result.state.candidate_hash
        seen = evaluated
    checks = {
        "state_assertions_passed": True,
        "additions_were_unevaluated": additions_are_new,
        "trace_observations_are_cumulative": trace_is_cumulative,
        "candidate_hash_is_stable": candidate_hash_stable,
        "unique_model_calls_match_ids": result.state.model_call_count == len(result.state.evaluated_global_ids),
        "unique_model_calls_match_coordinate_hashes": (
            result.state.model_call_count == len(result.state.evaluated_coordinate_hashes)
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"benchmark identity validation failed: {checks}")
    return checks


def run_scenario(scenario):
    candidate, test = frozen_benchmark_inputs()
    profile = _profile_for(scenario)
    runner_class = _ScheduledCVRunner if scenario == "overfit_fallback" else AdaptiveSparsePCE
    model = (lambda xi: np.full(xi.shape[1], np.nan)) if scenario == "runtime_failure" else _benchmark_model
    accuracy = (lambda fit, qoi: False) if scenario in {"max_order", "overfit_fallback"} else None
    order_max = 1 if scenario == "runtime_failure" else (3 if scenario != "converged" else 2)
    runner = runner_class(
        candidate, _hermite_vandermonde, model, profile, order_min=1, order_max=order_max,
        test_xi=test, qoi=np.mean, accuracy=accuracy,
        doi_centers=test[:, :4], doi_radius=0.55, criterion="S", random_state=BENCHMARK_SEED,
    )
    result = runner.run()
    checks = _validate_identity(result)
    stages = Counter(item.stage for item in result.trace)
    return {
        "scenario": scenario,
        "status": result.status,
        "stop_reason": result.stop_reason,
        "profile": asdict(profile),
        "trigger_disclosure": (
            "real LARS/CV fits with manifest-only order CV labels {1:1,2:2,3:3}"
            if scenario == "overfit_fallback" else None
        ),
        "candidate_hash": array_hash(candidate),
        "test_hash": array_hash(test),
        "trace_hash": result.trace_hash(),
        "trace_rows": len(result.trace),
        "stage_counts": dict(sorted(stages.items())),
        "evaluated_global_ids": list(result.state.evaluated_global_ids),
        "evaluated_coordinate_hashes": sorted(result.state.evaluated_coordinate_hashes),
        "model_call_count": result.state.model_call_count,
        "final_polynomial_order": result.state.polynomial_order,
        "order_cv_history": [list(item) for item in result.state.order_cv_history],
        "outer_qoi_history": list(result.state.outer_qoi_history),
        "invariant_checks": checks,
        "trace": [asdict(item) for item in result.trace],
    }


def run_suite(scenarios=None):
    scenarios = list(scenarios or ("converged", "max_order", "overfit_fallback", "runtime_failure"))
    candidate, test = frozen_benchmark_inputs()
    manifest = {
        "schema_version": 1,
        "benchmark": BENCHMARK_NAME,
        "input": {
            "rng": "numpy.random.Generator(PCG64)", "seed": BENCHMARK_SEED,
            "cv_seed": CV_SEED, "candidate_shape": list(candidate.shape),
            "test_shape": list(test.shape), "candidate_hash": array_hash(candidate),
            "test_hash": array_hash(test),
        },
        "git": _git_metadata(),
        "dependencies": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "reproduce_command": REPRODUCE_COMMAND,
        "scenarios": {name: run_scenario(name) for name in scenarios},
    }
    stable_payload = dict(manifest)
    # Git provenance is reported but excluded from the content hash so that
    # committing an otherwise identical benchmark does not change the hash.
    stable_payload.pop("git")
    manifest["stable_manifest_hash"] = hashlib.sha256(
        json.dumps(stable_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["all", "converged", "max_order",
                                               "overfit_fallback", "runtime_failure"], default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    scenarios = None if args.scenario == "all" else [args.scenario]
    manifest = run_suite(scenarios)
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
