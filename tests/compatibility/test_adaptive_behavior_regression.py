import numpy as np
import pytest

import uqra
from uqra.adaptive import (AdaptiveSparsePCE, array_hash,
                           canonical_legacy_lars_trace, compare_doi,
                           compare_oed_rounds, compare_preprocessing,
                           freeze_hermite_inputs, literal_index_bug_trace,
                           literal_legacy_profile, modern_lars_trace)


def hermite_factory(order):
    return uqra.Hermite(d=2, deg=order, hem_type="probabilists")


@pytest.fixture(scope="module")
def frozen():
    return freeze_hermite_inputs(hermite_factory)


def test_shared_canonical_hermite_inputs_are_frozen(frozen):
    repeated = freeze_hermite_inputs(hermite_factory)
    assert not frozen.candidate_xi.flags.writeable
    assert not frozen.test_xi.flags.writeable
    assert frozen.candidate_hash == repeated.candidate_hash
    assert frozen.test_hash == repeated.test_hash
    assert frozen.train_hash == repeated.train_hash
    np.testing.assert_array_equal(frozen.candidate_xi, repeated.candidate_xi)
    np.testing.assert_array_equal(frozen.test_xi, repeated.test_xi)


def test_weight_and_preprocessing_arrays_match_elementwise(frozen):
    X = hermite_factory(3).vandermonde(frozen.candidate_xi[:, frozen.train_global_ids])
    trace = compare_preprocessing(X, frozen.y_train, frozen.sample_weight)
    np.testing.assert_array_equal(trace["legacy_X"], trace["modern_X"])
    np.testing.assert_array_equal(trace["legacy_y"], trace["modern_y"])
    np.testing.assert_array_equal(trace["offset"], np.zeros(X.shape[1]))
    np.testing.assert_array_equal(trace["scale"], np.ones(X.shape[1]))


def test_lars_entry_cv_path_and_truncation_match(frozen):
    X = hermite_factory(3).vandermonde(frozen.candidate_xi[:, frozen.train_global_ids])
    legacy = canonical_legacy_lars_trace(X, frozen.y_train, frozen.sample_weight,
                                         n_splits=6, cv_seed=frozen.cv_seed)
    modern = modern_lars_trace(X, frozen.y_train, frozen.sample_weight,
                               n_splits=6, cv_seed=frozen.cv_seed)
    assert legacy.fold_test_ids == modern.fold_test_ids
    assert legacy.active_path == modern.active_path
    np.testing.assert_allclose(legacy.cv_path, modern.cv_path, rtol=0, atol=1e-15)
    assert legacy.selected_path_length == modern.selected_path_length
    assert legacy.selected_active_ids == modern.selected_active_ids


@pytest.mark.parametrize("criterion", ["D", "S"])
def test_rrqr_scores_and_each_selected_id_match_canonical(frozen, criterion):
    X = hermite_factory(2).vandermonde(frozen.candidate_xi)
    legacy_design = uqra.OptimalDesign(X)
    trace = compare_oed_rounds(X, legacy_design, criterion=criterion, n_rounds=3)
    assert len(trace.initial_ids) == X.shape[1]
    assert len(trace.selected_ids) == 3
    for legacy_scores, modern_scores, candidates, selected in zip(
            trace.legacy_scores_by_round, trace.modern_scores_by_round,
            trace.candidate_ids_by_round, trace.selected_ids):
        np.testing.assert_allclose(legacy_scores, modern_scores, rtol=2e-12, atol=2e-12)
        scores = np.asarray(modern_scores)
        assert selected == candidates[int(np.flatnonzero(scores == np.max(scores))[0])]


def test_doi_candidates_mapping_and_literal_index_differences(frozen):
    evaluated = list(frozen.train_global_ids[:6])
    unevaluated = [i for i in range(frozen.candidate_xi.shape[1]) if i not in set(evaluated)]
    test_y = 1.5 + 0.7 * frozen.test_xi[0] - 0.35 * (frozen.test_xi[1] ** 2 - 1.0)
    doi = compare_doi(frozen.candidate_xi, frozen.test_xi, test_y, y0=1.5,
                      unevaluated_ids=unevaluated, n_centers=4, radius=0.9)
    assert doi.legacy_global_ids == doi.modern_global_ids
    assert tuple(sorted(doi.modern_local_to_global)) == doi.modern_global_ids
    assert set(doi.modern_global_ids).issubset(unevaluated)

    first = np.arange(frozen.candidate_xi.shape[1])
    second = np.roll(first, 7)
    local_ids = [1, 3]
    chosen_local = [0, min(2, len(doi.modern_local_to_global) - 1)]
    bug = literal_index_bug_trace(
        frozen.candidate_xi, first_permutation=first, second_permutation=second,
        retained_local_ids=local_ids, doi_global_ids=doi.modern_local_to_global,
        doi_local_ids=chosen_local,
    )
    assert bug["first_coordinate_hashes"] != bug["legacy_second_coordinate_hashes"]
    assert bug["first_global_ids"] == bug["modern_second_global_ids"]
    assert bug["legacy_doi_appended_ids"] != bug["modern_doi_appended_ids"]
    assert bug["classification"] == {
        "cross_order": "IDX-01 defect correction", "doi": "IDX-02 defect correction"}


def test_cumulative_observations_and_literal_stop_position_match_declared_behavior(frozen):
    def vander(order, xi):
        return hermite_factory(order).vandermonde(xi)
    profile = literal_legacy_profile(cv_folds=4, cv_seed=frozen.cv_seed,
                                     max_inner_iterations=1)
    model = lambda xi: 1.5 + 0.7 * xi[0] - 0.35 * (xi[1] ** 2 - 1.0)
    runner = AdaptiveSparsePCE(
        frozen.candidate_xi, vander, model, profile, order_min=1, order_max=3,
        test_xi=frozen.test_xi, qoi=np.mean, criterion="S",
    )
    result = runner.run()
    completed = [record for record in result.trace if record.stage == "order_completed"]
    assert [record.order for record in completed] == [1, 2, 3]
    assert all(set(previous.evaluated_global_ids).issubset(current.evaluated_global_ids)
               for previous, current in zip(completed, completed[1:]))
    assert result.status == "completed"
    assert result.stop_reason == "literal_orders_completed"
    assert result.trace[-1].order == 3
    assert result.state.model_call_count == len(result.state.evaluated_global_ids)
    assert all(record.candidate_hash == frozen.candidate_hash for record in result.trace)
