"""Canonical-legacy versus modern adaptive-PCE behavior regression helpers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from sklearn.linear_model import Lars, LinearRegression
from sklearn.model_selection import KFold

from .doi import build_doi
from .optimal_design import optimality_scores, rrqr_initial_ids
from .sparse_pce import fit_lars_path, legacy_preprocess
from .state import array_hash, coordinate_hash


@dataclass(frozen=True)
class FrozenHermiteInputs:
    candidate_xi: np.ndarray
    test_xi: np.ndarray
    train_global_ids: tuple[int, ...]
    y_train: np.ndarray
    sample_weight: np.ndarray
    candidate_hash: str
    test_hash: str
    train_hash: str
    seed: int
    cv_seed: int


@dataclass(frozen=True)
class LarsBehaviorTrace:
    active_path: tuple[int, ...]
    cv_path: tuple[float, ...]
    selected_active_ids: tuple[int, ...]
    selected_path_length: int
    fold_test_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class OEDRoundTrace:
    criterion: str
    initial_ids: tuple[int, ...]
    candidate_ids_by_round: tuple[tuple[int, ...], ...]
    legacy_scores_by_round: tuple[tuple[float, ...], ...]
    modern_scores_by_round: tuple[tuple[float, ...], ...]
    selected_ids: tuple[int, ...]


@dataclass(frozen=True)
class DOIBehaviorTrace:
    center_test_ids: tuple[int, ...]
    legacy_global_ids: tuple[int, ...]
    modern_global_ids: tuple[int, ...]
    modern_local_to_global: tuple[int, ...]


def freeze_hermite_inputs(hermite_factory, *, seed=90210, cv_seed=73,
                          n_candidate=48, n_test=72, n_train=18, order=3):
    rng = np.random.default_rng(seed)
    candidate = rng.normal(size=(2, n_candidate))
    test = rng.normal(size=(2, n_test))
    train_ids = tuple(int(i) for i in rng.choice(n_candidate, size=n_train, replace=False))
    model = lambda xi: 1.5 + 0.7 * xi[0] - 0.35 * (xi[1] ** 2 - 1.0) + 0.15 * xi[0] * xi[1]
    y = model(candidate[:, train_ids])
    hermite = hermite_factory(order)
    X = hermite.vandermonde(candidate[:, train_ids])
    # Positive nonuniform weights exercise the canonical sqrt(w) rescaling path.
    weights = 0.75 + np.sum(X ** 2, axis=1) / (2.0 * X.shape[1])
    train_payload = np.column_stack((np.asarray(train_ids), y, weights))
    candidate.setflags(write=False); test.setflags(write=False)
    y.setflags(write=False); weights.setflags(write=False)
    return FrozenHermiteInputs(
        candidate, test, train_ids, y, weights, array_hash(candidate), array_hash(test),
        array_hash(train_payload), int(seed), int(cv_seed),
    )


def canonical_legacy_weight_trace(X, y, sample_weight):
    """Translate canonical PCE._rescale_data exactly without sparse allocation."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    sqrt_weight = np.sqrt(np.asarray(sample_weight, dtype=float).reshape(-1))
    return sqrt_weight, X * sqrt_weight[:, None], y * sqrt_weight


def compare_preprocessing(X, y, sample_weight):
    legacy_sqrt, legacy_X, legacy_y = canonical_legacy_weight_trace(X, y, sample_weight)
    modern = legacy_preprocess(X, y, sample_weight, fit_intercept=False, normalize=False)
    return {
        "sqrt_weight": legacy_sqrt,
        "legacy_X": legacy_X,
        "modern_X": modern.X,
        "legacy_y": legacy_y,
        "modern_y": modern.y,
        "offset": modern.offset,
        "scale": modern.scale,
    }


def _fixed_folds(n_samples, n_splits, seed):
    splitter = KFold(min(n_splits, n_samples), shuffle=True, random_state=seed)
    return list(splitter.split(np.arange(n_samples)))


def canonical_legacy_lars_trace(X, y, sample_weight, *, n_splits, cv_seed):
    """Version-compatible translation of canonical OLSLAR with fixed fold identity."""
    _, WX, Wy = canonical_legacy_weight_trace(X, y, sample_weight)
    lars = Lars(fit_intercept=False).fit(WX, Wy)
    active = [int(i) for i in lars.active_]
    folds = _fixed_folds(len(Wy), n_splits, cv_seed)
    cv_path = []
    for length in range(1, len(active) + 1):
        columns = active[:length]
        fold_errors = []
        for train, test in folds:
            ols = LinearRegression(fit_intercept=False).fit(WX[train][:, columns], Wy[train])
            residual = Wy[test] - ols.predict(WX[test][:, columns])
            fold_errors.append(float(np.mean(residual ** 2)))
        cv_path.append(float(np.mean(fold_errors)))
    selected_length = int(np.argmin(cv_path)) + 1
    return LarsBehaviorTrace(tuple(active), tuple(cv_path), tuple(active[:selected_length]),
                             selected_length, tuple(tuple(int(i) for i in test) for _, test in folds))


def modern_lars_trace(X, y, sample_weight, *, n_splits, cv_seed):
    fit = fit_lars_path(X, y, sample_weight, n_splits=n_splits, shuffle=True,
                        random_state=cv_seed)
    folds = _fixed_folds(len(y), n_splits, cv_seed)
    return LarsBehaviorTrace(tuple(fit.active_path), tuple(fit.cv_path),
                             tuple(fit.selected_active_ids), len(fit.selected_active_ids),
                             tuple(tuple(int(i) for i in test) for _, test in folds))


def compare_oed_rounds(X, legacy_design, *, criterion, n_rounds=3):
    """Compare canonical full-rank incremental formulas after shared RRQR."""
    X = np.asarray(X, dtype=float)
    criterion = criterion.upper()
    initial_legacy = tuple(int(i) for i in legacy_design._initial_samples_rrqr(X.shape[1]))
    initial_modern = tuple(rrqr_initial_ids(X, X.shape[1]))
    if initial_legacy != initial_modern:
        raise AssertionError("canonical and modern RRQR initial IDs differ")
    selected = list(initial_modern)
    candidate_rounds, legacy_rounds, modern_rounds, chosen = [], [], [], []
    for _ in range(n_rounds):
        candidates = [i for i in range(X.shape[0]) if i not in set(selected)]
        A, B = X[selected], X[candidates]
        base_logdet = float(np.linalg.slogdet(A.T @ A)[1])
        if criterion == "D":
            incremental = np.asarray(legacy_design._greedy_update_D_Optimality_full(A, B))
            legacy_scores = base_logdet + np.log(incremental)
        elif criterion == "S":
            incremental = np.asarray(legacy_design._greedy_update_S_Optimality_full(A, B))
            legacy_scores = base_logdet + incremental
        else:
            raise ValueError("criterion must be D or S")
        modern_scores = np.asarray(optimality_scores(A, B, criterion))
        best = int(np.flatnonzero(modern_scores == np.max(modern_scores))[0])
        selected_id = int(candidates[best])
        candidate_rounds.append(tuple(candidates))
        legacy_rounds.append(tuple(float(v) for v in legacy_scores))
        modern_rounds.append(tuple(float(v) for v in modern_scores))
        chosen.append(selected_id); selected.append(selected_id)
    return OEDRoundTrace(criterion, initial_modern, tuple(candidate_rounds),
                         tuple(legacy_rounds), tuple(modern_rounds), tuple(chosen))


def compare_doi(candidate_xi, test_xi, test_y, *, y0, unevaluated_ids,
                n_centers=4, radius=0.65):
    center_ids = np.argsort(np.abs(np.asarray(test_y) - y0), kind="stable")[:n_centers]
    centers = np.asarray(test_xi)[:, center_ids]
    unevaluated = [int(i) for i in unevaluated_ids]
    legacy_ids = set()
    for center in centers.T:
        distances = np.linalg.norm(np.asarray(candidate_xi)[:, unevaluated] - center[:, None], axis=0)
        local = np.flatnonzero(distances < radius)
        legacy_ids.update(unevaluated[int(i)] for i in local)
    modern = build_doi(candidate_xi, unevaluated, centers, radius, minimum_size=0, fallback="skip")
    return DOIBehaviorTrace(tuple(int(i) for i in center_ids), tuple(sorted(legacy_ids)),
                            tuple(sorted(modern.global_ids)), tuple(modern.local_to_global))


def literal_index_bug_trace(candidate_xi, *, first_permutation, second_permutation,
                            retained_local_ids, doi_global_ids, doi_local_ids):
    """Record canonical IDX-01/IDX-02 literal behavior beside corrected identity."""
    xi = np.asarray(candidate_xi)
    retained = [int(i) for i in retained_local_ids]
    first_global = [int(first_permutation[i]) for i in retained]
    legacy_second_global = [int(second_permutation[i]) for i in retained]
    modern_second_global = list(first_global)
    legacy_doi_appended = [int(i) for i in doi_local_ids]
    modern_doi_appended = [int(doi_global_ids[i]) for i in doi_local_ids]
    payload = {
        "first_global_ids": first_global,
        "legacy_second_global_ids": legacy_second_global,
        "modern_second_global_ids": modern_second_global,
        "first_coordinate_hashes": [coordinate_hash(xi[:, i]) for i in first_global],
        "legacy_second_coordinate_hashes": [coordinate_hash(xi[:, i]) for i in legacy_second_global],
        "legacy_doi_appended_ids": legacy_doi_appended,
        "modern_doi_appended_ids": modern_doi_appended,
        "classification": {"cross_order": "IDX-01 defect correction", "doi": "IDX-02 defect correction"},
    }
    payload["trace_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload
