"""Deterministic RRQR and UQRA D/S-optimal incremental design."""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def rrqr_initial_ids(X: np.ndarray, n: int, candidate_ids=None) -> list[int]:
    X = np.asarray(X, dtype=float)
    ids = np.arange(X.shape[0]) if candidate_ids is None else np.asarray(candidate_ids, dtype=int)
    if n < 0 or n > min(len(ids), X.shape[1]):
        raise ValueError("invalid RRQR design size")
    _, _, pivots = qr(X[ids].T, pivoting=True, mode="economic")
    return [int(ids[i]) for i in pivots[:n]]


def optimality_scores(selected: np.ndarray, candidates: np.ndarray, criterion: str) -> np.ndarray:
    A = np.asarray(selected, dtype=float)
    B = np.asarray(candidates, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("selected and candidates must have equal column counts")
    criterion = criterion.upper()
    if criterion not in {"D", "S"}:
        raise ValueError("criterion must be D or S")
    p = A.shape[1]
    scores = np.empty(B.shape[0])
    for index, row in enumerate(B):
        updated = np.vstack((A, row))
        # Canonical UQRA TSM phase uses the first k+1 columns until the
        # selected design reaches full column rank.
        objective_matrix = updated[:, :updated.shape[0]] if updated.shape[0] < p else updated
        gram = objective_matrix.T @ objective_matrix
        sign, logdet = np.linalg.slogdet(gram)
        d_score = logdet if sign > 0 else -np.inf
        if criterion == "D":
            scores[index] = d_score
        else:
            norms_sq = np.sum(objective_matrix ** 2, axis=0)
            scores[index] = d_score - np.sum(np.log(norms_sq)) if np.all(norms_sq > 0) else -np.inf
    return scores


def greedy_optimal_ids(X: np.ndarray, n_add: int, *, selected_ids=None,
                       candidate_ids=None, criterion="S", initialize_rrqr=True) -> list[int]:
    X = np.asarray(X, dtype=float)
    selected = [] if selected_ids is None else [int(i) for i in selected_ids]
    if len(selected) != len(set(selected)):
        raise ValueError("selected_ids contains duplicates")
    candidates = ([i for i in range(X.shape[0]) if i not in set(selected)]
                  if candidate_ids is None else [int(i) for i in candidate_ids if int(i) not in set(selected)])
    if n_add < 0 or n_add > len(candidates):
        raise ValueError("invalid requested design increment")
    added = []
    if initialize_rrqr and not selected and n_add:
        initial_count = min(n_add, X.shape[1], len(candidates))
        initial = rrqr_initial_ids(X, initial_count, candidates)
        selected.extend(initial); added.extend(initial)
        candidates = [i for i in candidates if i not in set(initial)]
    while len(added) < n_add:
        scores = optimality_scores(X[selected], X[candidates], criterion)
        best = int(np.flatnonzero(scores == np.nanmax(scores))[0])
        chosen = candidates.pop(best)
        selected.append(chosen); added.append(chosen)
    return added
