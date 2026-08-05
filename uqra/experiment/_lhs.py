"""Latin-hypercube generator used internally by :mod:`uqra.experiment`.

The small implementation replaces the unmaintained pyDOE2 dependency while
preserving the criteria and ``RandomState`` sequence used by UQRA's public LHS
wrapper.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist


def _random_state(value):
    if value is None:
        return np.random.RandomState()
    if isinstance(value, np.random.RandomState):
        return value
    return np.random.RandomState(value)


def _classic(dimensions, samples, rng):
    cut = np.linspace(0.0, 1.0, samples + 1)
    points = rng.rand(samples, dimensions) * np.diff(cut)[:, None] + cut[:-1, None]
    design = np.empty_like(points)
    for column in range(dimensions):
        design[:, column] = points[rng.permutation(samples), column]
    return design


def _centered(dimensions, samples, rng):
    centers = (np.arange(samples, dtype=float) + 0.5) / samples
    # Preserve pyDOE2's historical RNG consumption before its permutations.
    design = np.empty_like(rng.rand(samples, dimensions))
    for column in range(dimensions):
        design[:, column] = rng.permutation(centers)
    return design


def lhs(dimensions, samples=None, criterion=None, iterations=5, random_state=None):
    """Generate a pyDOE2-compatible Latin-hypercube design."""
    dimensions = int(dimensions)
    samples = dimensions if samples is None else int(samples)
    iterations = 5 if iterations is None else int(iterations)
    if dimensions < 1 or samples < 1 or iterations < 1:
        raise ValueError("dimensions, samples, and iterations must be positive")
    rng = _random_state(random_state)
    key = None if criterion is None else str(criterion).lower()
    aliases = {"c": "center", "m": "maximin", "cm": "centermaximin",
               "corr": "correlation"}
    key = aliases.get(key, key)
    if key not in (None, "center", "maximin", "centermaximin", "correlation"):
        raise ValueError('Invalid value for "criterion": {}'.format(criterion))
    if key is None:
        return _classic(dimensions, samples, rng)
    if key == "center":
        return _centered(dimensions, samples, rng)

    best = None
    best_score = -np.inf
    for _ in range(iterations):
        candidate = (_centered(dimensions, samples, rng)
                     if key == "centermaximin" else _classic(dimensions, samples, rng))
        if key in ("maximin", "centermaximin"):
            score = float(np.min(pdist(candidate))) if samples > 1 else np.inf
        else:
            correlation = np.corrcoef(candidate, rowvar=False)
            off_diagonal = correlation - np.eye(dimensions)
            score = -float(np.max(np.abs(off_diagonal)))
        if score > best_score:
            best, best_score = candidate.copy(), score
    return best
