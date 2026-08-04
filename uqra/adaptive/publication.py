"""Frozen publication protocol and deterministic sensitivity manifest."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
import sklearn

from .benchmark import (BENCHMARK_SEED, CV_SEED, _ScheduledCVRunner,
                        _benchmark_model, frozen_benchmark_inputs)
from .controller import AdaptiveSparsePCE
from .profiles import publication_profile
from .state import array_hash


PROTOCOL_VERSION = 1
REPRODUCE_COMMAND = (
    "python -m uqra.adaptive.publication "
    "--output artifacts/adaptive_phase11_publication_manifest.json"
)
BASE_PARAMETERS = {
    "order_min": 1, "order_max": 3, "criterion": "S",
    "order_budget_factor": 2.0, "cv_folds": 4, "cv_shuffle": True,
    "cv_seed": CV_SEED, "outer_stable_checks": 1, "qoi_tolerance": 0.05,
    "qoi_epsilon": 1e-12, "accuracy_max_cv_error": 1e-6,
    "validity": "all sample-identity invariants and finite QoI/CV",
    "minimum_doi_size": 4, "doi_fallback": "expand", "doi_radius": 0.55,
    "inner_qoi_tolerance": None, "inner_stable_checks": 1,
    "max_inner_iterations": 3, "overfit_rebuild": True,
    "candidate_seed": BENCHMARK_SEED, "controller_seed": BENCHMARK_SEED,
}
SENSITIVITY_CASES = {
    "baseline": {},
    "budget_low": {"order_budget_factor": 1.5},
    "budget_high": {"order_budget_factor": 2.5},
    "doi_global": {"doi_fallback": "global"},
    "doi_skip": {"doi_fallback": "skip"},
    "doi_radius_narrow": {"doi_radius": 0.35},
    "doi_radius_wide": {"doi_radius": 0.80},
    "cv_folds_5": {"cv_folds": 5},
    "outer_stable_2": {"outer_stable_checks": 2},
    "accuracy_strict": {"accuracy_max_cv_error": 1e-32},
    "overfit_fallback": {"overfit_schedule": {1: 1.0, 2: 2.0, 3: 3.0}},
}


def frozen_publication_protocol():
    candidate, test = frozen_benchmark_inputs()
    payload = {
        "schema": "uqra.adaptive.publication-protocol/v1",
        "protocol_version": PROTOCOL_VERSION,
        "implementations": {
            "canonical_uqra": "historical replay unavailable; evidence-only comparator",
            "modern_compatible": "AdaptiveSparsePCE with uqra.Hermite",
            "portable_control": "AdaptiveSparsePCE with independent NumPy Hermite-E evaluator",
        },
        "input": {
            "rng": "numpy.random.Generator(PCG64)", "seed": BENCHMARK_SEED,
            "candidate_shape": list(candidate.shape), "test_shape": list(test.shape),
            "candidate_hash": array_hash(candidate), "test_hash": array_hash(test),
        },
        "base_parameters": dict(BASE_PARAMETERS),
        "sensitivity_cases": {name: dict(values) for name, values in SENSITIVITY_CASES.items()},
        "overfit_reporting": "reported separately; never pooled with ordinary convergence",
        "reproduce_command": REPRODUCE_COMMAND,
    }
    payload["protocol_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _modern_vandermonde(order, xi):
    from uqra.polynomial.hermite import Hermite
    return Hermite(d=2, deg=order, hem_type="probabilists").vandermonde(xi)


def _portable_vandermonde(order, xi):
    xi = np.asarray(xi, dtype=float)
    degrees = [item for item in itertools.product(range(order + 1), repeat=2)
               if sum(item) <= order]
    degrees.sort(key=sum)
    one_dimensional = [np.polynomial.hermite_e.hermevander(row, order) for row in xi]
    matrix = np.ones((xi.shape[1], len(degrees)), dtype=float)
    for column, degree in enumerate(degrees):
        norm = 1
        for dimension, value in enumerate(degree):
            matrix[:, column] *= one_dimensional[dimension][:, value]
            norm *= math.factorial(value)
        matrix[:, column] /= math.sqrt(norm)
    return matrix


def _identity_checks(result):
    result.state.assert_invariants()
    seen = set()
    additions_new = True
    cumulative = True
    for row in result.trace:
        additions_new &= not bool(seen.intersection(row.added_global_ids))
        evaluated = set(row.evaluated_global_ids)
        cumulative &= seen.issubset(evaluated)
        seen = evaluated
    checks = {
        "state_invariants": True,
        "additions_were_unevaluated": additions_new,
        "trace_is_cumulative": cumulative,
        "unique_calls_match_ids": (
            result.state.model_call_count == len(result.state.evaluated_global_ids)
        ),
        "unique_calls_match_coordinates": (
            result.state.model_call_count == len(result.state.evaluated_coordinate_hashes)
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"publication validity failed: {checks}")
    return checks


def _run_case(name, overrides, implementation):
    parameters = dict(BASE_PARAMETERS)
    parameters.update(overrides)
    candidate, test = frozen_benchmark_inputs()
    vander = _modern_vandermonde if implementation == "modern_compatible" else _portable_vandermonde
    runner_type = _ScheduledCVRunner if "overfit_schedule" in parameters else AdaptiveSparsePCE
    profile = publication_profile(
        cv_folds=parameters["cv_folds"], cv_shuffle=parameters["cv_shuffle"],
        cv_seed=parameters["cv_seed"], order_budget_factor=parameters["order_budget_factor"],
        outer_stable_checks=parameters["outer_stable_checks"],
        qoi_tolerance=parameters["qoi_tolerance"], qoi_epsilon=parameters["qoi_epsilon"],
        minimum_doi_size=parameters["minimum_doi_size"], doi_fallback=parameters["doi_fallback"],
        inner_qoi_tolerance=parameters["inner_qoi_tolerance"],
        inner_stable_checks=parameters["inner_stable_checks"],
        max_inner_iterations=parameters["max_inner_iterations"],
        overfit_rebuild=parameters["overfit_rebuild"],
    )
    accuracy_limit = parameters["accuracy_max_cv_error"]
    accuracy = lambda fit, qoi: bool(np.isfinite(fit.cv_error) and fit.cv_error <= accuracy_limit)
    runner = runner_type(
        candidate, vander, _benchmark_model, profile,
        order_min=parameters["order_min"], order_max=parameters["order_max"],
        test_xi=test, qoi=np.mean, accuracy=accuracy, doi_centers=test[:, :4],
        doi_radius=parameters["doi_radius"], criterion=parameters["criterion"],
        random_state=parameters["controller_seed"],
    )
    result = runner.run()
    checks = _identity_checks(result)
    final_cv = None if result.model is None else float(result.model.cv_error)
    qoi = next((row.qoi for row in reversed(result.trace) if row.qoi is not None), None)
    return {
        "case": name, "implementation": implementation,
        "overrides": overrides, "profile": asdict(profile),
        "status": result.status, "stop_reason": result.stop_reason,
        "is_overfit_fallback": result.status == "overfit_fallback",
        "final_cv_error": final_cv, "final_qoi": qoi,
        "accuracy_passed": bool(final_cv is not None and final_cv <= accuracy_limit),
        "validity_passed": all(checks.values()), "invariant_checks": checks,
        "model_call_count": result.state.model_call_count,
        "evaluated_global_ids": list(result.state.evaluated_global_ids),
        "trace_hash": result.trace_hash(),
        "stage_counts": dict(sorted(Counter(row.stage for row in result.trace).items())),
        "order_cv_history": [list(item) for item in result.state.order_cv_history],
        "outer_qoi_history": list(result.state.outer_qoi_history),
    }


def _git_metadata():
    root = Path(__file__).resolve().parents[2]
    def run(*arguments):
        process = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True,
                                 text=True, check=False)
        return process.stdout.strip() if process.returncode == 0 else None
    dirty = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "worktree_dirty": None if dirty is None else bool(dirty)}


def _source_tree_hash():
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "uqra" / "adaptive").glob("*.py"))
    sources.append(root / "uqra" / "polynomial" / "hermite.py")
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_publication_suite():
    protocol = frozen_publication_protocol()
    modern = {name: _run_case(name, overrides, "modern_compatible")
              for name, overrides in SENSITIVITY_CASES.items()}
    portable = {name: _run_case(name, overrides, "portable_control")
                for name, overrides in SENSITIVITY_CASES.items()}
    comparisons = {}
    for name in SENSITIVITY_CASES:
        comparisons[name] = {
            "status_equal": modern[name]["status"] == portable[name]["status"],
            "stop_reason_equal": modern[name]["stop_reason"] == portable[name]["stop_reason"],
            "selected_ids_equal": modern[name]["evaluated_global_ids"] == portable[name]["evaluated_global_ids"],
            "trace_hash_equal": modern[name]["trace_hash"] == portable[name]["trace_hash"],
            "qoi_abs_difference": abs(modern[name]["final_qoi"] - portable[name]["final_qoi"])
                if modern[name]["final_qoi"] is not None and portable[name]["final_qoi"] is not None else None,
        }
    candidate, _ = frozen_benchmark_inputs()
    basis_error = max(float(np.max(np.abs(_modern_vandermonde(order, candidate)
                                           - _portable_vandermonde(order, candidate))))
                      for order in range(1, BASE_PARAMETERS["order_max"] + 1))
    results = {
        "canonical_uqra": {
            "status": "unavailable", "result_kind": "evidence_only",
            "reason": "canonical FourBranch inputs, RNG state, and per-round archive unavailable",
            "allowed_evidence": ["phase7 canonical kernel fixture", "phase9 archive inventory"],
            "prohibited_claim": "historical final Pf reproduction",
        },
        "modern_compatible": modern,
        "portable_control": portable,
    }
    manifest = {
        "schema": "uqra.adaptive.publication-manifest/v1", "protocol": protocol,
        "git": _git_metadata(), "source_tree_hash": _source_tree_hash(),
        "dependencies": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "results": results, "modern_vs_portable": comparisons,
        "portable_basis_max_abs_error": basis_error,
        "overfit_fallback_counts": {
            "canonical_uqra": None,
            "modern_compatible": sum(row["is_overfit_fallback"] for row in modern.values()),
            "portable_control": sum(row["is_overfit_fallback"] for row in portable.values()),
        },
        "reproduce_command": REPRODUCE_COMMAND,
    }
    stable = dict(manifest); stable.pop("git")
    manifest["stable_manifest_hash"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    manifest = run_publication_suite()
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
