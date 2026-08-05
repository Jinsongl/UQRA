"""New deterministic Ishigami reduced software fixture, not paper production."""
from __future__ import annotations

import numpy as np

from .reduced_fixture import run_reduced_fixture
from .state import array_hash

BENCHMARK_NAME = "ishigami_reduced_v1"
SCENARIO = "reduced"
SEEDS = {"candidate": 26080601, "test": 26080602, "reference": 26080603}
CV_SEED = 26080604
SIZES = {"candidate": 256, "test": 384, "reference": 4096}
INPUT_HASHES = {
    "candidate": "dcd094b440fa280c273ebf12987c045c71095872de3ba21042db923d8c2a4c6a",
    "test": "ee18a46c313537de96e183054a596eb51d8af04729f8123d4422ec0dbb6e93c3",
    "reference": "2e28f08fc184d407199ff0b98e4804fe38dc217bf066f7a95e09a759f280dff0",
}
EXPECTED_CONTRACT_TRACE_HASH = "697fb5aa0a321fa871d59befd11571719cfa614cfc5517c4cfd76ce4d134a91c"
EXPECTED_CONTRACT_MANIFEST_HASH = "37ebf3d45c94534eae6d8c51c366d4c5983aa33eec6eb55752529a93bdcdaad7"
EXPECTED_REFERENCE_VARIANCE = 13.814374748854757


def ishigami(xi):
    xi = np.asarray(xi, dtype=float).reshape(3, -1)
    x = np.pi * xi
    return np.sin(x[0]) + 7.0 * np.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * np.sin(x[0])


def _generate_inputs():
    arrays = {name: np.random.default_rng(SEEDS[name]).uniform(-1.0, 1.0, size=(3, size))
              for name, size in SIZES.items()}
    for array in arrays.values(): array.setflags(write=False)
    return arrays


def frozen_inputs():
    arrays = _generate_inputs()
    actual = {name: array_hash(array) for name, array in arrays.items()}
    if actual != INPUT_HASHES: raise RuntimeError(f"Ishigami frozen input hash mismatch: {actual}")
    return arrays


def _vandermonde(order, xi):
    from uqra.polynomial.legendre import Legendre
    return Legendre(d=3, deg=order).vandermonde(xi)


def run_suite(scenarios=None):
    if list(scenarios or (SCENARIO,)) != [SCENARIO]: raise ValueError("Ishigami supports only 'reduced'")
    arrays = frozen_inputs()
    return run_reduced_fixture(
        name=BENCHMARK_NAME, arrays=arrays, seeds=SEEDS, cv_seed=CV_SEED,
        model=ishigami, vandermonde=_vandermonde, qoi=np.var,
        doi_score=lambda xi: -np.abs((np.pi * xi[2]) ** 4 * np.sin(np.pi * xi[0])),
        input_contract={"distribution": "independent Uniform[-pi,pi]", "a": 7.0, "b": 0.1,
                        "purpose": "nonlinearity_and_x1_x3_interaction"},
        reference_metric=lambda ref: {"name": "output_variance", "value": float(np.var(ishigami(ref)))},
    )
