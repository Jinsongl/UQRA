"""Two-level adaptive sparse-PCE controller."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import numpy as np

from .doi import build_doi
from .optimal_design import greedy_optimal_ids
from .profiles import CompatibilityProfile
from .sparse_pce import SparsePCEFit, fit_lars_path
from .state import AdaptiveState


STOP_REASONS = {"outer_qoi_converged", "max_order_reached", "overfit_rebuild_converged",
                "overfit_rebuild_not_converged", "runtime_failure"}


@dataclass
class RoundTrace:
    order: int
    stage: str
    evaluated_global_ids: list[int]
    active_path: list[int]
    selected_active_ids: list[int]
    cv_path: list[float]
    added_global_ids: list[int] = field(default_factory=list)
    doi_global_ids: list[int] = field(default_factory=list)
    doi_local_ids: list[int] = field(default_factory=list)
    qoi: float | None = None


@dataclass
class AdaptiveResult:
    status: str
    stop_reason: str
    state: AdaptiveState
    model: SparsePCEFit | None
    trace: list[RoundTrace]

    def trace_hash(self) -> str:
        payload = json.dumps([asdict(item) for item in self.trace], sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()


class AdaptiveSparsePCE:
    def __init__(self, candidate_xi, vandermonde, model, profile: CompatibilityProfile, *,
                 order_min, order_max, test_xi=None, qoi=None, accuracy=None, doi_centers=None,
                 doi_radius=0.0, criterion="S", random_state=0):
        profile.validate_for_run()
        self.state = AdaptiveState(candidate_xi)
        self.vandermonde = vandermonde
        self.model_fn = model
        self.profile = profile
        self.order_min, self.order_max = int(order_min), int(order_max)
        self.test_xi, self.qoi, self.accuracy = test_xi, qoi, accuracy
        self.doi_centers, self.doi_radius = doi_centers, float(doi_radius)
        self.criterion, self.random_state = criterion, int(random_state)
        self.trace: list[RoundTrace] = []
        self._models: dict[int, SparsePCEFit] = {}
        self._outer_stable_count = 0

    def _fit(self, order, X) -> SparsePCEFit:
        ids = self.state.evaluated_global_ids
        y = np.asarray([self.state.y_by_global_id[i] for i in ids])
        fit = fit_lars_path(X[ids], y, n_splits=self.profile.cv_folds,
                            shuffle=self.profile.cv_shuffle, random_state=self.profile.cv_seed)
        self.state.active_path = list(fit.active_path)
        self.state.selected_active_ids = list(fit.selected_active_ids)
        self.state.cv_path = list(fit.cv_path)
        self._models[order] = fit
        return fit

    def _qoi(self, fit, order):
        if self.qoi is None:
            return None
        Xtest = np.asarray(self.vandermonde(order, self.test_xi), dtype=float)
        value = float(self.qoi(fit.predict(Xtest)))
        if not np.isfinite(value):
            raise FloatingPointError("non-finite QoI")
        return value

    def _record(self, order, stage, fit, *, added=None, doi=None, local=None, qoi=None):
        self.trace.append(RoundTrace(order, stage, list(self.state.evaluated_global_ids),
                                    list(fit.active_path), list(fit.selected_active_ids), list(fit.cv_path),
                                    list(added or []), list(doi or []), list(local or []), qoi))

    def run(self) -> AdaptiveResult:
        last_fit = None
        try:
            previous_sparsity = self.profile.initial_sparsity or 0
            for order in range(self.order_min, self.order_max + 1):
                self.state.polynomial_order = order
                X = np.asarray(self.vandermonde(order, self.state.candidate_xi), dtype=float)
                self.state.full_basis_ids = list(range(X.shape[1]))
                target = int(np.ceil(max(previous_sparsity, 0.8 * X.shape[1])))
                budget = min(len(self.state.global_ids), int(np.ceil(self.profile.order_budget_factor * X.shape[1])))
                target = min(target, budget)
                gap = max(0, target - len(self.state.evaluated_global_ids))
                if gap:
                    added = greedy_optimal_ids(X, gap, selected_ids=self.state.evaluated_global_ids,
                                               candidate_ids=self.state.unevaluated_global_ids(), criterion=self.criterion)
                    self.state.evaluate_new(added, self.model_fn)
                    self.state.global_added_ids = list(added)
                if len(self.state.evaluated_global_ids) < 2:
                    raise RuntimeError("insufficient observations for LARS/CV")
                last_fit = self._fit(order, X)
                qoi_value = self._qoi(last_fit, order)
                self._record(order, "initial_fit", last_fit, added=self.state.global_added_ids, qoi=qoi_value)

                batch = min(self.profile.batch_size(len(last_fit.selected_active_ids)),
                            max(0, budget - len(self.state.evaluated_global_ids)))
                if batch:
                    global_added = greedy_optimal_ids(X[:, last_fit.selected_active_ids], batch,
                        selected_ids=self.state.evaluated_global_ids, candidate_ids=self.state.unevaluated_global_ids(),
                        criterion=self.criterion, initialize_rrqr=False)
                    self.state.evaluate_new(global_added, self.model_fn)
                    self.state.global_added_ids = list(global_added)
                    last_fit = self._fit(order, X)
                    self._record(order, "global_refit", last_fit, added=global_added, qoi=self._qoi(last_fit, order))

                if self.doi_centers is not None and len(self.state.unevaluated_global_ids()):
                    centers = self.doi_centers(last_fit, order) if callable(self.doi_centers) else self.doi_centers
                    doi = build_doi(self.state.candidate_xi, self.state.unevaluated_global_ids(), centers,
                                    self.doi_radius, minimum_size=self.profile.minimum_doi_size or 0,
                                    fallback=self.profile.doi_fallback)
                    self.state.doi_global_ids = list(doi.global_ids)
                    self.state.doi_local_to_global = list(doi.local_to_global)
                    n_local = min(self.profile.batch_size(len(last_fit.selected_active_ids)), len(doi.global_ids),
                                  max(0, budget - len(self.state.evaluated_global_ids)))
                    if n_local:
                        local_rows = greedy_optimal_ids(X[doi.global_ids][:, last_fit.selected_active_ids], n_local,
                                                       criterion=self.criterion)
                        global_ids = self.state.map_doi_local_ids(local_rows)
                        self.state.evaluate_new(global_ids, self.model_fn)
                        self.state.doi_added_ids = list(global_ids)
                        last_fit = self._fit(order, X)
                        self._record(order, "doi_refit", last_fit, added=global_ids, doi=doi.global_ids,
                                     local=local_rows, qoi=self._qoi(last_fit, order))

                qoi_value = self._qoi(last_fit, order)
                if qoi_value is not None:
                    self.state.outer_qoi_history.append(qoi_value)
                self.state.order_cv_history.append((order, float(last_fit.cv_error)))
                previous_sparsity = len(last_fit.selected_active_ids)
                if (self.profile.name == "publication" and self.profile.overfit_rebuild
                        and not self.state.overfit_rebuild_performed
                        and len(self.state.order_cv_history) >= 3):
                    recent = self.state.order_cv_history[-3:]
                    if recent[0][1] < recent[1][1] < recent[2][1]:
                        return self._overfit_rebuild(order)
                if self.profile.name == "publication" and len(self.state.outer_qoi_history) >= 2:
                    old, new = self.state.outer_qoi_history[-2:]
                    delta = abs(new - old) / max(abs(old), self.profile.qoi_epsilon)
                    accurate = True if self.accuracy is None else bool(self.accuracy(last_fit, new))
                    if delta <= self.profile.qoi_tolerance and accurate:
                        self._outer_stable_count += 1
                    else:
                        self._outer_stable_count = 0
                    if self._outer_stable_count >= self.profile.outer_stable_checks:
                        return self._finish("converged", "outer_qoi_converged", last_fit)
                self.state.assert_invariants()
            return self._finish("nonconverged", "max_order_reached", last_fit)
        except Exception:
            self.state.stop_reason = "runtime_failure"
            raise

    def _overfit_rebuild(self, trigger_order):
        rebuild_order = trigger_order - 1
        self.state.overfit_detected = True
        self.state.overfit_trigger_order = trigger_order
        self.state.overfit_rebuild_performed = True
        self.state.overfit_rebuild_sample_ids = list(self.state.evaluated_global_ids)
        X = np.asarray(self.vandermonde(rebuild_order, self.state.candidate_xi), dtype=float)
        rebuilt = self._fit(rebuild_order, X)
        rebuilt_qoi = self._qoi(rebuilt, rebuild_order)
        self._record(rebuild_order, "overfit_rebuild", rebuilt, qoi=rebuilt_qoi)
        finite = np.isfinite(rebuilt.cv_error) and (rebuilt_qoi is None or np.isfinite(rebuilt_qoi))
        converged = False
        if finite and rebuilt_qoi is not None and self.state.outer_qoi_history:
            reference = self.state.outer_qoi_history[-2] if len(self.state.outer_qoi_history) >= 2 else self.state.outer_qoi_history[-1]
            delta = abs(rebuilt_qoi - reference) / max(abs(reference), self.profile.qoi_epsilon)
            accurate = True if self.accuracy is None else bool(self.accuracy(rebuilt, rebuilt_qoi))
            converged = delta <= self.profile.qoi_tolerance and accurate
        if converged:
            return self._finish("converged_after_rebuild", "overfit_rebuild_converged", rebuilt)
        return self._finish("overfit_fallback", "overfit_rebuild_not_converged", rebuilt)

    def _finish(self, status, reason, model):
        if reason not in STOP_REASONS:
            raise ValueError("unknown stop reason")
        self.state.stop_reason = reason
        return AdaptiveResult(status, reason, self.state, model, self.trace)
