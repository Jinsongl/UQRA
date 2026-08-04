"""Modern, deterministic LARS path selection with legacy preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Lars, LinearRegression
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class PreprocessedData:
    X: np.ndarray
    y: np.ndarray
    offset: np.ndarray
    scale: np.ndarray
    y_offset: float


@dataclass
class SparsePCEFit:
    active_path: list[int]
    selected_active_ids: list[int]
    canonical_active_ids: list[int]
    cv_path: list[float]
    coefficients: np.ndarray
    intercept: float
    cv_error: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, self.selected_active_ids] @ self.coefficients + self.intercept


def legacy_preprocess(X, y, sample_weight=None, *, fit_intercept=False, normalize=False):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X and y shapes are incompatible")
    if sample_weight is None:
        sw = np.ones(y.size)
    else:
        sw = np.asarray(sample_weight, dtype=float).reshape(-1)
        if sw.size != y.size or np.any(sw < 0):
            raise ValueError("invalid sample_weight")
    sqrt_w = np.sqrt(sw)
    if fit_intercept:
        total = sw.sum()
        if total <= 0:
            raise ValueError("sample weights must have positive sum")
        offset = np.average(X, axis=0, weights=sw)
        y_offset = float(np.average(y, weights=sw))
    else:
        offset = np.zeros(X.shape[1])
        y_offset = 0.0
    centered = X - offset
    if normalize:
        scale = np.sqrt(np.sum(sw[:, None] * centered ** 2, axis=0))
        scale[scale == 0] = 1.0
    else:
        scale = np.ones(X.shape[1])
    return PreprocessedData(
        X=(centered / scale) * sqrt_w[:, None],
        y=(y - y_offset) * sqrt_w,
        offset=offset, scale=scale, y_offset=y_offset,
    )


def _folds(n_samples, n_splits, shuffle, seed):
    if n_samples < 2:
        raise ValueError("at least two observations are required")
    count = min(max(2, int(n_splits)), n_samples)
    return list(KFold(count, shuffle=shuffle, random_state=seed if shuffle else None).split(np.arange(n_samples)))


def fit_lars_path(X, y, sample_weight=None, *, fit_intercept=False, normalize=False,
                  n_splits=5, shuffle=True, random_state=0) -> SparsePCEFit:
    data = legacy_preprocess(X, y, sample_weight, fit_intercept=fit_intercept, normalize=normalize)
    lars = Lars(fit_intercept=False).fit(data.X, data.y)
    active_path = [int(i) for i in lars.active_]
    if not active_path:
        active_path = [int(np.argmax(np.abs(data.X.T @ data.y)))]
    folds = _folds(len(data.y), n_splits, shuffle, random_state)
    cv_path = []
    for length in range(1, len(active_path) + 1):
        cols = active_path[:length]
        errors = []
        for train, test in folds:
            model = LinearRegression(fit_intercept=False).fit(data.X[train][:, cols], data.y[train])
            residual = data.y[test] - model.predict(data.X[test][:, cols])
            errors.append(float(np.mean(residual ** 2)))
        cv_path.append(float(np.mean(errors)))
    # np.argmin is the declared deterministic earliest-prefix tie rule.
    selected = active_path[:int(np.argmin(cv_path)) + 1]
    final = LinearRegression(fit_intercept=False).fit(data.X[:, selected], data.y)
    scaled_coef = np.asarray(final.coef_, dtype=float) / data.scale[selected]
    intercept = data.y_offset - float(data.offset[selected] @ scaled_coef)
    if not fit_intercept:
        intercept = 0.0
    return SparsePCEFit(
        active_path=active_path,
        selected_active_ids=list(selected),
        canonical_active_ids=sorted(selected),
        cv_path=cv_path,
        coefficients=scaled_coef,
        intercept=intercept,
        cv_error=cv_path[len(selected) - 1],
    )
