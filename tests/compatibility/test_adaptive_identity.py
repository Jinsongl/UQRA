import numpy as np
import pytest

from uqra.adaptive import AdaptiveState, build_doi


def fixture_candidates():
    return np.array([[-2., -1., 0., 1., 2.], [0., 1., 2., 3., 4.]])


def test_candidate_pool_is_frozen_and_observations_accumulate():
    state = AdaptiveState(fixture_candidates())
    initial_hash = state.candidate_hash
    state.evaluate_new([3, 1], lambda x: x[0] + 2 * x[1])
    state.polynomial_order = 3
    state.evaluate_new([4], lambda x: x[0] + 2 * x[1])
    x, y = state.training_data()
    assert state.candidate_hash == initial_hash
    assert state.evaluated_global_ids == [3, 1, 4]
    np.testing.assert_array_equal(x, fixture_candidates()[:, [3, 1, 4]])
    np.testing.assert_array_equal(y, [7., 1., 10.])
    assert state.model_call_count == 3


def test_duplicate_global_id_and_duplicate_coordinate_are_guarded():
    state = AdaptiveState(np.array([[0., 0., 1.], [1., 1., 2.]]))
    state.evaluate_new([0], lambda x: np.ones(x.shape[1]))
    with pytest.raises(ValueError, match="already been evaluated"):
        state.evaluate_new([0], lambda x: np.ones(x.shape[1]))
    with pytest.raises(ValueError, match="coordinate"):
        state.evaluate_new([1], lambda x: np.ones(x.shape[1]))


def test_doi_local_rows_are_mapped_to_global_ids():
    state = AdaptiveState(fixture_candidates())
    state.evaluate_new([0, 2], lambda x: x[0])
    doi = build_doi(state.candidate_xi, state.unevaluated_global_ids(), np.array([[1.], [3.]]), 1.1)
    state.doi_global_ids = doi.global_ids
    state.doi_local_to_global = doi.local_to_global
    assert doi.global_ids == [3]
    assert state.map_doi_local_ids([0]) == [3]
