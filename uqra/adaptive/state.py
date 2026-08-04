"""Stable candidate identity and cumulative observation state."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Callable

import numpy as np


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def coordinate_hash(value: np.ndarray) -> str:
    return array_hash(np.asarray(value, dtype=np.float64).reshape(-1))


@dataclass
class AdaptiveState:
    """Mutable run state whose candidate identity is immutable."""

    candidate_xi: np.ndarray
    polynomial_order: int = 0
    candidate_hash: str = field(init=False)
    global_ids: np.ndarray = field(init=False)
    evaluated_global_ids: list[int] = field(default_factory=list)
    evaluated_coordinate_hashes: set[str] = field(default_factory=set)
    y_by_global_id: dict[int, float] = field(default_factory=dict)
    model_call_count: int = 0
    full_basis_ids: list[int] = field(default_factory=list)
    active_path: list[int] = field(default_factory=list)
    selected_active_ids: list[int] = field(default_factory=list)
    global_added_ids: list[int] = field(default_factory=list)
    doi_global_ids: list[int] = field(default_factory=list)
    doi_local_to_global: list[int] = field(default_factory=list)
    doi_added_ids: list[int] = field(default_factory=list)
    inner_qoi_history: list[float] = field(default_factory=list)
    outer_qoi_history: list[float] = field(default_factory=list)
    cv_path: list[float] = field(default_factory=list)
    order_cv_history: list[tuple[int, float]] = field(default_factory=list)
    overfit_detected: bool = False
    overfit_trigger_order: int | None = None
    overfit_rebuild_performed: bool = False
    overfit_rebuild_sample_ids: list[int] = field(default_factory=list)
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        xi = np.array(self.candidate_xi, dtype=np.float64, copy=True, ndmin=2)
        xi.setflags(write=False)
        self.candidate_xi = xi
        self.candidate_hash = array_hash(xi)
        self.global_ids = np.arange(xi.shape[1], dtype=np.int64)

    def unevaluated_global_ids(self) -> list[int]:
        evaluated = set(self.evaluated_global_ids)
        return [int(i) for i in self.global_ids if int(i) not in evaluated]

    def evaluate_new(self, global_ids: list[int], model: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        requested = [int(i) for i in global_ids]
        if len(requested) != len(set(requested)):
            raise ValueError("duplicate global_id in evaluation request")
        existing = set(self.evaluated_global_ids)
        if existing.intersection(requested):
            raise ValueError("a requested global_id has already been evaluated")
        if any(i < 0 or i >= self.candidate_xi.shape[1] for i in requested):
            raise IndexError("global_id outside frozen candidate pool")
        hashes = [coordinate_hash(self.candidate_xi[:, i]) for i in requested]
        if self.evaluated_coordinate_hashes.intersection(hashes) or len(hashes) != len(set(hashes)):
            raise ValueError("a requested coordinate has already been evaluated")
        if not requested:
            return np.empty(0, dtype=float)
        values = np.asarray(model(self.candidate_xi[:, requested]), dtype=float).reshape(-1)
        if values.size != len(requested):
            raise ValueError("model must return one scalar response per candidate")
        if not np.all(np.isfinite(values)):
            raise ValueError("model returned a non-finite response")
        for global_id, coord_hash, value in zip(requested, hashes, values):
            self.evaluated_global_ids.append(global_id)
            self.evaluated_coordinate_hashes.add(coord_hash)
            self.y_by_global_id[global_id] = float(value)
        self.model_call_count += len(requested)
        self.assert_invariants()
        return values

    def training_data(self) -> tuple[np.ndarray, np.ndarray]:
        ids = self.evaluated_global_ids
        return self.candidate_xi[:, ids], np.asarray([self.y_by_global_id[i] for i in ids])

    def map_doi_local_ids(self, local_ids: list[int]) -> list[int]:
        if len(local_ids) != len(set(local_ids)):
            raise ValueError("duplicate DoI local id")
        try:
            mapped = [int(self.doi_local_to_global[int(i)]) for i in local_ids]
        except IndexError as error:
            raise IndexError("DoI local id outside local-to-global mapping") from error
        before = set(self.unevaluated_global_ids())
        if not set(mapped).issubset(before):
            raise ValueError("DoI selection contains an evaluated global_id")
        return mapped

    def assert_invariants(self) -> None:
        if array_hash(self.candidate_xi) != self.candidate_hash:
            raise AssertionError("frozen candidate pool changed")
        if len(self.evaluated_global_ids) != len(set(self.evaluated_global_ids)):
            raise AssertionError("evaluated_global_ids contains duplicates")
        if set(self.evaluated_global_ids) != set(self.y_by_global_id):
            raise AssertionError("responses and evaluated global IDs differ")
        if self.model_call_count != len(self.evaluated_coordinate_hashes):
            raise AssertionError("model call and unique coordinate counts differ")
        if len(self.evaluated_global_ids) != len(self.evaluated_coordinate_hashes):
            raise AssertionError("two global IDs identify the same coordinate")
