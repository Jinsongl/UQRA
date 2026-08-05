import numpy as np

from uqra.adaptive import AdaptiveSparsePCE, publication_profile


def hermite_like_vandermonde(order, xi):
    x, y = xi
    columns = [np.ones_like(x)]
    for degree in range(1, order + 1):
        for x_degree in range(degree, -1, -1):
            columns.append(x ** x_degree * y ** (degree - x_degree))
    return np.column_stack(columns)


def run_fixture():
    rng = np.random.default_rng(72819)
    candidate = rng.normal(size=(2, 40))
    test = rng.normal(size=(2, 60))
    model = lambda x: 1. + x[0] + .5 * x[1] ** 2
    profile = publication_profile(cv_folds=4, cv_seed=41, qoi_tolerance=1e-12,
                                  outer_stable_checks=1, minimum_doi_size=3)
    runner = AdaptiveSparsePCE(candidate, hermite_like_vandermonde, model, profile,
        order_min=1, order_max=2, test_xi=test, qoi=lambda y: np.mean(y),
        doi_centers=np.zeros((2, 1)), doi_radius=.35, criterion="S")
    return runner.run()


def test_minimal_two_dimensional_fixture_is_identity_safe_and_repeatable():
    first = run_fixture()
    second = run_fixture()
    first.state.assert_invariants()
    assert first.trace_hash() == second.trace_hash()
    assert first.state.model_call_count == len(first.state.evaluated_global_ids)
    for record in first.trace:
        assert record.selected_active_ids == record.active_path[:len(record.selected_active_ids)]
    doi_records = [record for record in first.trace if record.stage == "doi_refit"]
    assert doi_records
    for record in doi_records:
        assert record.added_global_ids == [record.doi_global_ids[i] for i in record.doi_local_ids]


def test_publication_profile_records_finite_terminal_reason():
    result = run_fixture()
    assert result.stop_reason in {"outer_qoi_converged", "max_order_reached",
                                  "overfit_rebuild_converged", "overfit_rebuild_not_converged"}
    assert result.state.stop_reason == result.stop_reason
