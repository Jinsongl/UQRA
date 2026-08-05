import numpy as np

import uqra
from uqra.adaptive import AdaptiveSparsePCE, array_hash, publication_profile


def hermite_vandermonde(order, xi):
    return uqra.Hermite(d=2, deg=order, hem_type="probabilists").vandermonde(xi)


def test_canonical_hermite_basis_order_and_normalized_values():
    hermite = uqra.Hermite(d=2, deg=2, hem_type="probabilists")
    xi = np.array([[0.0, 1.0, -1.0], [0.0, 2.0, 1.0]])
    assert hermite.basis_degree == [(0, 0), (0, 1), (1, 0),
                                    (0, 2), (1, 1), (2, 0)]
    expected = np.array([
        [1.0, 0.0, 0.0, -1 / np.sqrt(2), 0.0, -1 / np.sqrt(2)],
        [1.0, 2.0, 1.0, 3 / np.sqrt(2), 2.0, 0.0],
        [1.0, 1.0, -1.0, 0.0, -1.0, 0.0],
    ])
    np.testing.assert_allclose(hermite.vandermonde(xi), expected, rtol=0, atol=1e-14)


def run_hermite_fixture():
    rng = np.random.default_rng(314159)
    candidate = rng.normal(size=(2, 50))
    test = rng.normal(size=(2, 80))
    profile = publication_profile(cv_folds=4, cv_seed=2718, inner_qoi_tolerance=None,
                                  max_inner_iterations=2, overfit_rebuild=False,
                                  outer_stable_checks=1, qoi_tolerance=1e-12)
    result = AdaptiveSparsePCE(
        candidate, hermite_vandermonde,
        lambda xi: 1.0 + xi[0] + 0.5 * (xi[1] ** 2 - 1.0), profile,
        order_min=1, order_max=2, test_xi=test, qoi=np.mean, criterion="S",
    ).run()
    return candidate, test, result


def test_real_hermite_fixture_is_repeatable_and_identity_safe():
    candidate1, test1, first = run_hermite_fixture()
    candidate2, test2, second = run_hermite_fixture()
    assert array_hash(candidate1) == array_hash(candidate2)
    assert array_hash(test1) == array_hash(test2)
    assert first.trace_hash() == second.trace_hash()
    assert first.status != "runtime_failure"
    first.state.assert_invariants()
    assert all(item.candidate_hash == first.state.candidate_hash for item in first.trace)
