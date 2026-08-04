import sys

import numpy as np
from scipy import stats

import uqra
from uqra.experiment._lhs import lhs


def test_uqra_import_and_lhs_do_not_load_pydoe2():
    assert "pyDOE2" not in sys.modules
    design = uqra.LHS([stats.uniform(), stats.norm()], criterion="maximin", iterations=4)
    first = design.samples(size=12, random_state=1234)
    second = design.samples(size=12, random_state=1234)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 12)


def test_each_dimension_occupies_every_latin_stratum():
    samples = 17
    design = lhs(3, samples=samples, criterion=None, random_state=81)
    strata = np.floor(design * samples).astype(int)
    for column in range(design.shape[1]):
        np.testing.assert_array_equal(np.sort(strata[:, column]), np.arange(samples))


def test_center_and_optimization_criteria_remain_supported():
    centered = lhs(2, samples=9, criterion="center", random_state=5)
    np.testing.assert_allclose(
        np.mod(centered * 9, 1.0), np.full_like(centered, 0.5), atol=1e-14)
    for criterion in ("maximin", "centermaximin", "correlation"):
        result = lhs(3, samples=10, criterion=criterion, iterations=3, random_state=5)
        assert result.shape == (10, 3)
        assert np.all((result > 0.0) & (result < 1.0))


def test_centered_seed_keeps_canonical_pydoe2_sequence():
    expected = np.array([
        [0.1, 0.9], [0.5, 0.5], [0.7, 0.1], [0.3, 0.7], [0.9, 0.3],
    ])
    np.testing.assert_array_equal(
        lhs(2, samples=5, criterion="center", random_state=42), expected)
