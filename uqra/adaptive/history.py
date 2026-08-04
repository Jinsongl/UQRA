"""Read-only identity and literal-index diagnostics for archived experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .regression import literal_index_bug_trace


def archived_array_identity(path):
    """Return file and NumPy identity without modifying the archived input."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    array = np.load(source, allow_pickle=False, mmap_mode="r")
    return {
        "source": str(source),
        "status": "recovered",
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "size_bytes": int(source.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def historical_literal_index_diagnostic(candidate_xi, *, seed=20260804,
                                        sample_size=100_000, retained_count=66,
                                        doi_size=256, doi_selected_count=32):
    """Quantify IDX-01/IDX-02 on an archived pool with declared replay choices.

    This is a source-semantics diagnostic, not a reconstruction of a historical
    run: archived permutations, selected IDs, and RNG state are required for the
    latter and must never be inferred here.
    """
    xi = np.asarray(candidate_xi)
    if xi.ndim != 2:
        raise ValueError("candidate_xi must have shape (dimension, samples)")
    n_samples = xi.shape[1]
    if not 0 < sample_size <= n_samples:
        raise ValueError("sample_size must be within the archived pool")
    rng = np.random.default_rng(seed)
    first = rng.choice(n_samples, size=sample_size, replace=False)
    second = rng.choice(n_samples, size=sample_size, replace=False)
    retained_count = min(int(retained_count), sample_size)
    retained_local = np.sort(rng.choice(sample_size, size=retained_count, replace=False))

    # A deterministic near-origin set supplies a real archived-coordinate DoI.
    radii = np.sum(xi[:, second] ** 2, axis=0)
    doi_local_in_second = np.argsort(radii, kind="stable")[:min(doi_size, sample_size)]
    doi_global = second[doi_local_in_second]
    chosen_local = np.arange(min(doi_selected_count, doi_global.size), dtype=int)
    trace = literal_index_bug_trace(
        xi, first_permutation=first, second_permutation=second,
        retained_local_ids=retained_local, doi_global_ids=doi_global,
        doi_local_ids=chosen_local,
    )

    first_global = np.asarray(trace["first_global_ids"], dtype=int)
    legacy_second = np.asarray(trace["legacy_second_global_ids"], dtype=int)
    second_lookup = set(int(value) for value in second)
    overlap = sorted(set(int(value) for value in first_global) & second_lookup)
    literal_excluded = set(int(value) for value in second[retained_local])
    correct_excluded = set(overlap)
    idx01 = {
        "retained_ids": int(first_global.size),
        "coordinate_identity_mismatches": int(np.count_nonzero(first_global != legacy_second)),
        "false_exclusions": len(literal_excluded - correct_excluded),
        "missed_prior_samples": len(correct_excluded - literal_excluded),
        "prior_samples_present_in_next_pool": len(correct_excluded),
    }
    literal_doi = np.asarray(trace["legacy_doi_appended_ids"], dtype=int)
    modern_doi = np.asarray(trace["modern_doi_appended_ids"], dtype=int)
    idx02 = {
        "selected_ids": int(modern_doi.size),
        "global_identity_mismatches": int(np.count_nonzero(literal_doi != modern_doi)),
        "literal_ids": literal_doi.tolist(),
        "stable_global_ids": modern_doi.tolist(),
    }
    payload = {
        "kind": "source_semantics_diagnostic_not_historical_replay",
        "seed": int(seed),
        "pool_shape": [int(value) for value in xi.shape],
        "sample_size_per_order": int(sample_size),
        "idx01": idx01,
        "idx02": idx02,
        "literal_trace_hash": trace["trace_hash"],
        "modern_stable_id_result": {
            "cross_order_coordinate_identity_mismatches": 0,
            "doi_global_identity_mismatches": 0,
        },
    }
    payload["diagnostic_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload
