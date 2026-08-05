from collections import Counter

import numpy as np
import pytest

import uqra.adaptive.controller as controller_module
from uqra.adaptive import (AdaptiveSparsePCE, SparsePCEFit,
                           literal_legacy_profile, publication_profile)


def polynomial_vandermonde(order, xi):
    x, y = xi
    columns = [np.ones_like(x)]
    for degree in range(1, order + 1):
        for x_degree in range(degree, -1, -1):
            columns.append(x ** x_degree * y ** (degree - x_degree))
    return np.column_stack(columns)


def fixture_inputs(n_candidate=80):
    rng = np.random.default_rng(20260804)
    return rng.normal(size=(2, n_candidate)), rng.normal(size=(2, 40))


def runner(profile, *, order_max=3, model=None, qoi=lambda values: np.mean(values), accuracy=None):
    candidate, test = fixture_inputs()
    return AdaptiveSparsePCE(
        candidate, polynomial_vandermonde,
        model or (lambda xi: 2.0 + xi[0] + 0.25 * xi[1] ** 2), profile,
        order_min=1, order_max=order_max, test_xi=test, qoi=qoi,
        accuracy=accuracy, criterion="S",
    )


def scripted_fit(X, y, **kwargs):
    cv_error = float(X.shape[1])
    return SparsePCEFit(
        active_path=[0], selected_active_ids=[0], canonical_active_ids=[0],
        cv_path=[cv_error], coefficients=np.array([0.0]), intercept=1.0,
        cv_error=cv_error,
    )


def test_multi_round_inner_loop_records_budget_and_refit_evidence():
    profile = publication_profile(order_budget_factor=6.0, inner_qoi_tolerance=None,
                                  max_inner_iterations=3, overfit_rebuild=False)
    result = runner(profile, order_max=1, qoi=None).run()
    global_refits = [item for item in result.trace if item.stage == "global_refit"]
    assert [item.inner_iteration for item in global_refits] == [1, 2, 3]
    assert [item.refit_count for item in global_refits] == sorted(item.refit_count for item in global_refits)
    assert all(item.evaluated_after > item.evaluated_before for item in global_refits)
    assert all(item.remaining_budget == item.order_budget - item.evaluated_after for item in global_refits)
    stopped = next(item for item in result.trace if item.stage == "inner_loop_stopped")
    assert stopped.inner_stop_reason == "max_inner_iterations"
    assert stopped.order_target < stopped.order_budget
    assert result.state.model_call_count == global_refits[-1].model_call_count


@pytest.mark.parametrize(
    ("case", "expected_status", "expected_reason"),
    [
        ("converged", "converged", "outer_qoi_converged"),
        ("max_order", "nonconverged", "max_order_reached"),
        ("rebuild_converged", "converged_after_rebuild", "overfit_rebuild_converged"),
        ("rebuild_fallback", "overfit_fallback", "overfit_rebuild_not_converged"),
        ("runtime_failure", "runtime_failure", "runtime_failure"),
        ("literal_complete", "completed", "literal_orders_completed"),
    ],
)
def test_six_terminal_outcomes(monkeypatch, case, expected_status, expected_reason):
    if case == "converged":
        profile = publication_profile(outer_stable_checks=1, qoi_tolerance=1e9,
                                      inner_qoi_tolerance=None, overfit_rebuild=False)
        result = runner(profile, order_max=2).run()
    elif case == "max_order":
        profile = publication_profile(inner_qoi_tolerance=None, overfit_rebuild=False)
        result = runner(profile, order_max=2, accuracy=lambda fit, qoi: False).run()
    elif case in {"rebuild_converged", "rebuild_fallback"}:
        monkeypatch.setattr(controller_module, "fit_lars_path", scripted_fit)
        profile = publication_profile(outer_stable_checks=99, qoi_tolerance=1e-12,
                                      inner_qoi_tolerance=None, max_inner_iterations=1)
        accurate = (lambda fit, qoi: case == "rebuild_converged")
        result = runner(profile, order_max=3, accuracy=accurate).run()
    elif case == "runtime_failure":
        profile = publication_profile(inner_qoi_tolerance=None)
        result = runner(profile, order_max=1,
                        model=lambda xi: np.full(xi.shape[1], np.nan)).run()
    else:
        profile = literal_legacy_profile(max_inner_iterations=1)
        result = runner(profile, order_max=2).run()
    assert (result.status, result.stop_reason) == (expected_status, expected_reason)
    assert result.trace[-1].transition_to == "terminated"
    assert result.state.stop_reason == expected_reason
    for previous, current in zip(result.trace, result.trace[1:]):
        assert current.transition_from == previous.transition_to
    if case.startswith("rebuild_"):
        rebuild = next(item for item in result.trace if item.stage == "overfit_rebuild")
        assert rebuild.rebuild_order == 2
        assert rebuild.rebuild_sample_ids == result.state.evaluated_global_ids
        assert rebuild.refit_count > 0
        assert result.state.polynomial_order == 2
    if case == "runtime_failure":
        assert result.trace[-1].error_type == "ValueError"
        assert "non-finite" in result.trace[-1].error_message


def test_trace_stage_matrix_covers_each_declared_transition(monkeypatch):
    monkeypatch.setattr(controller_module, "fit_lars_path", scripted_fit)
    profile = publication_profile(outer_stable_checks=99, inner_qoi_tolerance=None,
                                  max_inner_iterations=1)
    result = runner(profile, order_max=3, accuracy=lambda fit, qoi: False).run()
    stages = Counter(item.stage for item in result.trace)
    for required in ("order_started", "initial_design", "initial_fit", "global_refit",
                     "inner_loop_stopped", "order_completed", "overfit_detected",
                     "overfit_rebuild", "terminated"):
        assert stages[required] >= 1
