import numpy as np

from uqra.adaptive import (fit_lars_path, greedy_optimal_ids, legacy_preprocess,
                           optimality_scores, rrqr_initial_ids)


def test_legacy_preprocess_matches_weighted_center_and_l2_normalize():
    X = np.array([[1., 2.], [3., 4.], [5., 8.]])
    y = np.array([1., 3., 8.])
    w = np.array([1., 2., 1.])
    data = legacy_preprocess(X, y, w, fit_intercept=True, normalize=True)
    np.testing.assert_allclose(data.offset, np.average(X, axis=0, weights=w))
    np.testing.assert_allclose(data.y_offset, np.average(y, weights=w))
    np.testing.assert_allclose(np.sum(data.X * np.sqrt(w)[:, None], axis=0), 0., atol=1e-14)
    np.testing.assert_allclose(np.sum(data.X ** 2, axis=0), 1.)


def test_lars_path_is_preserved_separately_from_canonical_columns():
    x = np.linspace(-1., 1., 18)
    X = np.column_stack([np.ones_like(x), x, x ** 2, x ** 3])
    y = 3 * x ** 3 + .2 * x + .01 * np.sin(13 * x)
    first = fit_lars_path(X, y, n_splits=6, random_state=19)
    second = fit_lars_path(X, y, n_splits=6, random_state=19)
    assert first.active_path == second.active_path
    assert first.cv_path == second.cv_path
    assert first.selected_active_ids == first.active_path[:len(first.selected_active_ids)]
    assert first.canonical_active_ids == sorted(first.selected_active_ids)


def test_rrqr_and_greedy_ties_are_deterministic():
    X = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 1.]])
    assert rrqr_initial_ids(X, 2) == rrqr_initial_ids(X, 2)
    assert greedy_optimal_ids(X, 3, criterion="S") == greedy_optimal_ids(X, 3, criterion="S")


def test_uqra_s_score_is_logdet_minus_column_norm_product():
    A = np.array([[1., 0.], [0., 1.]])
    B = np.array([[1., 1.], [2., 0.]])
    actual = optimality_scores(A, B, "S")
    expected = []
    for row in B:
        updated = np.vstack([A, row])
        expected.append(np.linalg.slogdet(updated.T @ updated)[1]
                        - np.log(np.sum(updated[:, 0] ** 2))
                        - np.log(np.sum(updated[:, 1] ** 2)))
    np.testing.assert_allclose(actual, expected)


def test_underdetermined_score_uses_canonical_truncated_square_columns():
    A = np.array([[1., 2., 99.]])
    B = np.array([[3., 1., -99.], [2., 4., 500.]])
    scores = optimality_scores(A, B, "D")
    expected = [np.linalg.slogdet(np.vstack([A, row])[:, :2].T
                                  @ np.vstack([A, row])[:, :2])[1] for row in B]
    np.testing.assert_allclose(scores, expected)
