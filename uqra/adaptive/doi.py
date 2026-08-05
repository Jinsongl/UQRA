"""Domain-of-interest construction with explicit global identity."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DomainOfInterest:
    global_ids: list[int]
    local_to_global: list[int]
    radius_used: float
    fallback_used: str | None = None


def build_doi(candidate_xi, unevaluated_global_ids, centers, radius, *, minimum_size=0,
              fallback="skip", expansion_factor=2.0, max_expansions=8) -> DomainOfInterest:
    xi = np.asarray(candidate_xi, dtype=float)
    ids = np.asarray(unevaluated_global_ids, dtype=int)
    centers = np.asarray(centers, dtype=float)
    if centers.ndim == 1:
        centers = centers.reshape(xi.shape[0], -1)
    if centers.shape[0] != xi.shape[0] or radius < 0:
        raise ValueError("invalid DoI centers or radius")
    used = float(radius)
    distances = np.min(np.linalg.norm(xi[:, ids, None] - centers[:, None, :], axis=0), axis=1) if len(ids) else np.empty(0)
    selected = ids[distances <= used].tolist()
    fallback_used = None
    if len(selected) < minimum_size and fallback == "expand":
        fallback_used = "expand"
        for _ in range(max_expansions):
            used *= expansion_factor
            selected = ids[distances <= used].tolist()
            if len(selected) >= minimum_size:
                break
    if len(selected) < minimum_size and fallback in {"global", "legacy_nearest"}:
        fallback_used = fallback
        count = min(len(ids), minimum_size if fallback == "global" else max(100, minimum_size))
        order = np.argsort(distances, kind="stable")[:count]
        selected = ids[order].tolist()
    if len(selected) < minimum_size and fallback not in {"skip", "expand"}:
        raise ValueError("DoI minimum size could not be met")
    result = [int(i) for i in selected]
    return DomainOfInterest(result, list(result), used, fallback_used)
