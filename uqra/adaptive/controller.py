"""Two-level adaptive sparse-PCE controller with auditable transitions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import numpy as np

from .doi import build_doi
from .optimal_design import greedy_optimal_ids
from .profiles import CompatibilityProfile
from .sparse_pce import SparsePCEFit, fit_lars_path
from .state import AdaptiveState


class ControllerState(str, Enum):
    CREATED = "created"
    ORDER_STARTED = "order_started"
    INITIAL_DESIGN_COMPLETED = "initial_design_completed"
    INITIAL_FIT_COMPLETED = "initial_fit_completed"
    GLOBAL_REFIT_COMPLETED = "global_refit_completed"
    DOI_CONSTRUCTED = "doi_constructed"
    DOI_REFIT_COMPLETED = "doi_refit_completed"
    INNER_LOOP_STOPPED = "inner_loop_stopped"
    ORDER_COMPLETED = "order_completed"
    OVERFIT_DETECTED = "overfit_detected"
    OVERFIT_REBUILD_COMPLETED = "overfit_rebuild_completed"
    TERMINATED = "terminated"


STOP_REASONS = {
    "outer_qoi_converged", "max_order_reached", "literal_orders_completed",
    "overfit_rebuild_converged", "overfit_rebuild_not_converged", "runtime_failure",
}


@dataclass
class RoundTrace:
    order: int
    stage: str
    transition_from: str
    transition_to: str
    inner_iteration: int
    evaluated_global_ids: list[int]
    active_path: list[int]
    selected_active_ids: list[int]
    cv_path: list[float]
    added_global_ids: list[int] = field(default_factory=list)
    doi_global_ids: list[int] = field(default_factory=list)
    doi_local_ids: list[int] = field(default_factory=list)
    qoi: float | None = None
    order_target: int | None = None
    order_budget: int | None = None
    evaluated_before: int = 0
    evaluated_after: int = 0
    remaining_budget: int | None = None
    refit_count: int = 0
    model_call_count: int = 0
    candidate_hash: str = ""
    inner_stop_reason: str | None = None
    fallback_used: str | None = None
    rebuild_order: int | None = None
    rebuild_sample_ids: list[int] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class AdaptiveResult:
    status: str
    stop_reason: str
    state: AdaptiveState
    model: SparsePCEFit | None
    trace: list[RoundTrace]

    def trace_hash(self) -> str:
        payload = json.dumps([asdict(item) for item in self.trace], sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()


class AdaptiveSparsePCE:
    def __init__(self, candidate_xi, vandermonde, model, profile: CompatibilityProfile, *,
                 order_min, order_max, test_xi=None, qoi=None, accuracy=None, doi_centers=None,
                 doi_radius=0.0, criterion="S", random_state=0):
        profile.validate_for_run()
        if int(order_min) > int(order_max):
            raise ValueError("order_min must not exceed order_max")
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
        self._controller_state = ControllerState.CREATED
        self._refit_count = 0
        self._last_fit: SparsePCEFit | None = None

    def _fit(self, order, X) -> SparsePCEFit:
        ids = self.state.evaluated_global_ids
        y = np.asarray([self.state.y_by_global_id[i] for i in ids])
        fit = fit_lars_path(X[ids], y, n_splits=self.profile.cv_folds,
                            shuffle=self.profile.cv_shuffle, random_state=self.profile.cv_seed)
        self.state.active_path = list(fit.active_path)
        self.state.selected_active_ids = list(fit.selected_active_ids)
        self.state.cv_path = list(fit.cv_path)
        self._models[order] = fit
        self._last_fit = fit
        self._refit_count += 1
        self.state.assert_invariants()
        return fit

    def _qoi(self, fit, order):
        if self.qoi is None:
            return None
        Xtest = np.asarray(self.vandermonde(order, self.test_xi), dtype=float)
        value = float(self.qoi(fit.predict(Xtest)))
        if not np.isfinite(value):
            raise FloatingPointError("non-finite QoI")
        return value

    def _transition(self, to_state, order, stage, fit=None, *, iteration=0, added=None,
                    doi=None, local=None, qoi=None, target=None, budget=None, before=None,
                    inner_stop=None, fallback=None, rebuild_order=None, rebuild_ids=None,
                    error=None):
        previous = self._controller_state
        self._controller_state = ControllerState(to_state)
        evaluated_after = len(self.state.evaluated_global_ids)
        trace = RoundTrace(
            order=int(order), stage=stage, transition_from=previous.value,
            transition_to=self._controller_state.value, inner_iteration=int(iteration),
            evaluated_global_ids=list(self.state.evaluated_global_ids),
            active_path=list(fit.active_path) if fit is not None else list(self.state.active_path),
            selected_active_ids=list(fit.selected_active_ids) if fit is not None else list(self.state.selected_active_ids),
            cv_path=list(fit.cv_path) if fit is not None else list(self.state.cv_path),
            added_global_ids=list(added or []), doi_global_ids=list(doi or []),
            doi_local_ids=list(local or []), qoi=qoi, order_target=target, order_budget=budget,
            evaluated_before=evaluated_after if before is None else int(before),
            evaluated_after=evaluated_after,
            remaining_budget=None if budget is None else max(0, int(budget) - evaluated_after),
            refit_count=self._refit_count, model_call_count=self.state.model_call_count,
            candidate_hash=self.state.candidate_hash, inner_stop_reason=inner_stop,
            fallback_used=fallback, rebuild_order=rebuild_order,
            rebuild_sample_ids=list(rebuild_ids or []),
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        )
        self.trace.append(trace)
        return trace

    def _order_design(self, order, X, previous_sparsity):
        target = int(np.ceil(max(previous_sparsity, 0.8 * X.shape[1])))
        budget = min(len(self.state.global_ids), int(np.ceil(self.profile.order_budget_factor * X.shape[1])))
        target = min(target, budget)
        gap = min(max(0, target - len(self.state.evaluated_global_ids)),
                  len(self.state.unevaluated_global_ids()))
        before = len(self.state.evaluated_global_ids)
        added = []
        if gap:
            added = greedy_optimal_ids(X, gap, selected_ids=self.state.evaluated_global_ids,
                                       candidate_ids=self.state.unevaluated_global_ids(), criterion=self.criterion)
            self.state.evaluate_new(added, self.model_fn)
            self.state.global_added_ids = list(added)
        self._transition(ControllerState.INITIAL_DESIGN_COMPLETED, order, "initial_design",
                         added=added, target=target, budget=budget, before=before)
        return target, budget

    def _inner_loop(self, order, X, fit, target, budget):
        stable_count = 0
        previous_qoi = self._qoi(fit, order)
        iteration = 0
        while len(self.state.evaluated_global_ids) < budget and self.state.unevaluated_global_ids():
            iteration += 1
            if self.profile.max_inner_iterations is not None and iteration > self.profile.max_inner_iterations:
                reason = "max_inner_iterations"
                break
            iteration_added = 0
            batch = min(self.profile.batch_size(len(fit.selected_active_ids)),
                        budget - len(self.state.evaluated_global_ids),
                        len(self.state.unevaluated_global_ids()))
            if batch:
                before = len(self.state.evaluated_global_ids)
                global_added = greedy_optimal_ids(X[:, fit.selected_active_ids], batch,
                    selected_ids=self.state.evaluated_global_ids,
                    candidate_ids=self.state.unevaluated_global_ids(), criterion=self.criterion,
                    initialize_rrqr=False)
                self.state.evaluate_new(global_added, self.model_fn)
                self.state.global_added_ids = list(global_added)
                iteration_added += len(global_added)
                fit = self._fit(order, X)
                current_qoi = self._qoi(fit, order)
                self._transition(ControllerState.GLOBAL_REFIT_COMPLETED, order, "global_refit", fit,
                                 iteration=iteration, added=global_added, qoi=current_qoi,
                                 target=target, budget=budget, before=before)
                previous_qoi = current_qoi if previous_qoi is None else previous_qoi

            if (self.doi_centers is not None and self.state.unevaluated_global_ids()
                    and len(self.state.evaluated_global_ids) < budget):
                centers = self.doi_centers(fit, order) if callable(self.doi_centers) else self.doi_centers
                doi = build_doi(self.state.candidate_xi, self.state.unevaluated_global_ids(), centers,
                                self.doi_radius, minimum_size=self.profile.minimum_doi_size or 0,
                                fallback=self.profile.doi_fallback)
                self.state.doi_global_ids = list(doi.global_ids)
                self.state.doi_local_to_global = list(doi.local_to_global)
                self._transition(ControllerState.DOI_CONSTRUCTED, order, "doi_constructed", fit,
                                 iteration=iteration, doi=doi.global_ids, target=target, budget=budget,
                                 fallback=doi.fallback_used)
                n_local = min(self.profile.batch_size(len(fit.selected_active_ids)), len(doi.global_ids),
                              budget - len(self.state.evaluated_global_ids))
                if n_local:
                    before = len(self.state.evaluated_global_ids)
                    local_rows = greedy_optimal_ids(X[doi.global_ids][:, fit.selected_active_ids], n_local,
                                                   criterion=self.criterion)
                    global_ids = self.state.map_doi_local_ids(local_rows)
                    self.state.evaluate_new(global_ids, self.model_fn)
                    self.state.doi_added_ids = list(global_ids)
                    iteration_added += len(global_ids)
                    fit = self._fit(order, X)
                    current_qoi = self._qoi(fit, order)
                    self._transition(ControllerState.DOI_REFIT_COMPLETED, order, "doi_refit", fit,
                                     iteration=iteration, added=global_ids, doi=doi.global_ids,
                                     local=local_rows, qoi=current_qoi, target=target, budget=budget,
                                     before=before, fallback=doi.fallback_used)

            current_qoi = self._qoi(fit, order)
            if current_qoi is not None:
                self.state.inner_qoi_history.append(current_qoi)
            if (self.profile.inner_qoi_tolerance is not None and previous_qoi is not None
                    and current_qoi is not None):
                delta = abs(current_qoi - previous_qoi) / max(abs(previous_qoi), self.profile.qoi_epsilon)
                stable_count = stable_count + 1 if delta <= self.profile.inner_qoi_tolerance else 0
                previous_qoi = current_qoi
            if stable_count >= self.profile.inner_stable_checks:
                reason = "inner_qoi_stable"
                break
            if iteration_added == 0:
                reason = "no_candidates_added"
                break
        else:
            reason = "order_budget_reached" if len(self.state.evaluated_global_ids) >= budget else "candidate_pool_exhausted"
        self._transition(ControllerState.INNER_LOOP_STOPPED, order, "inner_loop_stopped", fit,
                         iteration=iteration, qoi=self._qoi(fit, order), target=target,
                         budget=budget, inner_stop=reason)
        return fit

    def run(self) -> AdaptiveResult:
        try:
            previous_sparsity = self.profile.initial_sparsity or 0
            for order in range(self.order_min, self.order_max + 1):
                self.state.polynomial_order = order
                self._transition(ControllerState.ORDER_STARTED, order, "order_started")
                X = np.asarray(self.vandermonde(order, self.state.candidate_xi), dtype=float)
                if X.ndim != 2 or X.shape[0] != self.state.candidate_xi.shape[1] or not np.all(np.isfinite(X)):
                    raise ValueError("Vandermonde matrix is invalid")
                self.state.full_basis_ids = list(range(X.shape[1]))
                target, budget = self._order_design(order, X, previous_sparsity)
                if len(self.state.evaluated_global_ids) < 2:
                    raise RuntimeError("insufficient observations for LARS/CV")
                fit = self._fit(order, X)
                self._transition(ControllerState.INITIAL_FIT_COMPLETED, order, "initial_fit", fit,
                                 qoi=self._qoi(fit, order), target=target, budget=budget)
                fit = self._inner_loop(order, X, fit, target, budget)
                qoi_value = self._qoi(fit, order)
                if qoi_value is not None:
                    self.state.outer_qoi_history.append(qoi_value)
                self.state.order_cv_history.append((order, float(fit.cv_error)))
                previous_sparsity = len(fit.selected_active_ids)
                self._transition(ControllerState.ORDER_COMPLETED, order, "order_completed", fit,
                                 qoi=qoi_value, target=target, budget=budget)

                if (self.profile.name == "publication" and self.profile.overfit_rebuild
                        and not self.state.overfit_rebuild_performed
                        and len(self.state.order_cv_history) >= 3):
                    recent = self.state.order_cv_history[-3:]
                    if recent[0][1] < recent[1][1] < recent[2][1]:
                        return self._overfit_rebuild(order)
                if self.profile.name == "publication" and len(self.state.outer_qoi_history) >= 2:
                    old, new = self.state.outer_qoi_history[-2:]
                    delta = abs(new - old) / max(abs(old), self.profile.qoi_epsilon)
                    accurate = True if self.accuracy is None else bool(self.accuracy(fit, new))
                    self._outer_stable_count = self._outer_stable_count + 1 if delta <= self.profile.qoi_tolerance and accurate else 0
                    if self._outer_stable_count >= self.profile.outer_stable_checks:
                        return self._finish("converged", "outer_qoi_converged", fit, order)
                self.state.assert_invariants()
            if self.profile.name == "literal_legacy":
                return self._finish("completed", "literal_orders_completed", self._last_fit, self.order_max)
            return self._finish("nonconverged", "max_order_reached", self._last_fit, self.order_max)
        except Exception as error:
            return self._runtime_failure(error)

    def _overfit_rebuild(self, trigger_order):
        rebuild_order = trigger_order - 1
        self.state.overfit_detected = True
        self.state.overfit_trigger_order = trigger_order
        self.state.overfit_rebuild_performed = True
        self.state.overfit_rebuild_sample_ids = list(self.state.evaluated_global_ids)
        self._transition(ControllerState.OVERFIT_DETECTED, trigger_order, "overfit_detected",
                         self._last_fit, rebuild_order=rebuild_order,
                         rebuild_ids=self.state.overfit_rebuild_sample_ids)
        self.state.polynomial_order = rebuild_order
        X = np.asarray(self.vandermonde(rebuild_order, self.state.candidate_xi), dtype=float)
        rebuilt = self._fit(rebuild_order, X)
        rebuilt_qoi = self._qoi(rebuilt, rebuild_order)
        self._transition(ControllerState.OVERFIT_REBUILD_COMPLETED, rebuild_order, "overfit_rebuild",
                         rebuilt, qoi=rebuilt_qoi, rebuild_order=rebuild_order,
                         rebuild_ids=self.state.overfit_rebuild_sample_ids)
        finite = np.isfinite(rebuilt.cv_error) and (rebuilt_qoi is None or np.isfinite(rebuilt_qoi))
        converged = False
        if finite and rebuilt_qoi is not None and self.state.outer_qoi_history:
            reference = self.state.outer_qoi_history[-2] if len(self.state.outer_qoi_history) >= 2 else self.state.outer_qoi_history[-1]
            delta = abs(rebuilt_qoi - reference) / max(abs(reference), self.profile.qoi_epsilon)
            accurate = True if self.accuracy is None else bool(self.accuracy(rebuilt, rebuilt_qoi))
            converged = delta <= self.profile.qoi_tolerance and accurate
        if converged:
            return self._finish("converged_after_rebuild", "overfit_rebuild_converged", rebuilt, rebuild_order)
        return self._finish("overfit_fallback", "overfit_rebuild_not_converged", rebuilt, rebuild_order)

    def _runtime_failure(self, error):
        self.state.stop_reason = "runtime_failure"
        order = self.state.polynomial_order or self.order_min
        self._transition(ControllerState.TERMINATED, order, "runtime_failure", self._last_fit,
                         error=error, rebuild_ids=self.state.overfit_rebuild_sample_ids)
        return AdaptiveResult("runtime_failure", "runtime_failure", self.state, self._last_fit, self.trace)

    def _finish(self, status, reason, model, order):
        if reason not in STOP_REASONS:
            raise ValueError("unknown stop reason")
        self.state.stop_reason = reason
        self._transition(ControllerState.TERMINATED, order, "terminated", model)
        return AdaptiveResult(status, reason, self.state, model, self.trace)
