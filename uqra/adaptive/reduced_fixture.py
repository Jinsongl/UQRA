"""Shared execution contract for deterministic reduced benchmark fixtures."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json

import numpy as np

from .controller import AdaptiveSparsePCE
from .profiles import publication_profile
from .state import array_hash


def dataset_identity(arrays, seeds):
    return {
        name: {"role": name, "rng": "numpy.random.Generator(PCG64)",
               "seed": seeds[name], "shape": list(array.shape),
               "sha256": array_hash(array)}
        for name, array in arrays.items()
    }


def run_reduced_fixture(*, name, arrays, seeds, cv_seed, model, vandermonde,
                        qoi, doi_score, input_contract, reference_metric,
                        order_min=2, order_max=4, doi_radius=0.75):
    candidate, test, reference = (arrays[key] for key in ("candidate", "test", "reference"))
    boundary_ids = np.argsort(np.asarray(doi_score(test)), kind="stable")[:8]
    profile = publication_profile(
        cv_folds=4, cv_seed=cv_seed, qoi_tolerance=1e-3,
        outer_stable_checks=1, inner_qoi_tolerance=None,
        max_inner_iterations=3, minimum_doi_size=8, overfit_rebuild=False,
    )
    runner = AdaptiveSparsePCE(
        candidate, vandermonde, model, profile, order_min=order_min,
        order_max=order_max, test_xi=test, qoi=qoi,
        doi_centers=test[:, boundary_ids], doi_radius=doi_radius,
        criterion="S", random_state=seeds["candidate"],
    )
    result = runner.run()
    result.state.assert_invariants()
    stages = Counter(row.stage for row in result.trace)
    scenario = {
        "scenario": "reduced", "status": result.status,
        "stop_reason": result.stop_reason,
        "candidate_hash": array_hash(candidate), "test_hash": array_hash(test),
        "reference_hash": array_hash(reference),
        "reference_metric": reference_metric(reference),
        "doi_center_test_ids": boundary_ids.tolist(),
        "trace_hash": result.trace_hash(), "trace_rows": len(result.trace),
        "stage_counts": dict(sorted(stages.items())),
        "model_call_count": result.state.model_call_count,
        "evaluated_global_ids": list(result.state.evaluated_global_ids),
        "evaluated_coordinate_hashes": sorted(result.state.evaluated_coordinate_hashes),
        "trace": [asdict(row) for row in result.trace],
    }
    manifest = {
        "schema_version": 1, "benchmark": name,
        "input": {"purpose": "software_benchmark", "scale": "reduced",
                  "historical_replay": False, "paper_production": False,
                  "cv_seed": cv_seed, "contract": input_contract,
                  "datasets": dataset_identity(arrays, seeds)},
        "scenarios": {"reduced": scenario},
    }
    manifest["stable_manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return manifest
