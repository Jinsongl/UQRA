"""New deterministic Gayton reduced reliability fixture, not paper production."""
from __future__ import annotations

import numpy as np

from .reduced_fixture import run_reduced_fixture
from .state import array_hash

BENCHMARK_NAME = "gayton_reduced_v1"
SCENARIO = "reduced"
SEEDS = {"candidate": 26080701, "test": 26080702, "reference": 26080703}
CV_SEED = 26080704
SIZES = {"candidate": 192, "test": 256, "reference": 4096}
INPUT_HASHES = {
    "candidate": "3a80fce33a0254e332cf4fd85ab758ecb8a370d112e06c89155572128c9da80f",
    "test": "4d202ea093be90c874f460b70762fbec84144174d18634e465b9b83c37e8f629",
    "reference": "d12f4e3cdf448c51d49e6194293c164250873cce72ae4609f6d928e397c169f3",
}
EXPECTED_TRACE_HASH = "3b49d8acd346e9b2f687e06393ed0bb6aae3ddf260826453b2f122078c99a3d0"
EXPECTED_MANIFEST_HASH = "5f660e4eba96214899350f81252b9858af2159bf36b22587eb70869bdc64b7ae"
EXPECTED_REFERENCE_FAILURE_PROBABILITY = 0.108642578125


def gayton_limit_state(xi):
    xi = np.asarray(xi, dtype=float).reshape(2, -1)
    x1, x2 = xi[0], xi[1] + 3.0
    return 0.5 * (x1 - 2.0) ** 2 - 1.5 * (x2 - 5.0) ** 3 - 3.0


def _generate_inputs():
    arrays = {name: np.random.default_rng(SEEDS[name]).normal(size=(2, size))
              for name, size in SIZES.items()}
    for array in arrays.values(): array.setflags(write=False)
    return arrays


def frozen_inputs():
    arrays = _generate_inputs()
    actual = {name: array_hash(array) for name, array in arrays.items()}
    if actual != INPUT_HASHES: raise RuntimeError(f"Gayton frozen input hash mismatch: {actual}")
    return arrays


def _vandermonde(order, xi):
    from uqra.polynomial.hermite import Hermite
    return Hermite(d=2, deg=order, hem_type="probabilists").vandermonde(xi)


def run_suite(scenarios=None):
    if list(scenarios or (SCENARIO,)) != [SCENARIO]: raise ValueError("Gayton supports only 'reduced'")
    arrays = frozen_inputs()
    return run_reduced_fixture(
        name=BENCHMARK_NAME, arrays=arrays, seeds=SEEDS, cv_seed=CV_SEED,
        model=gayton_limit_state, vandermonde=_vandermonde,
        qoi=lambda values: np.mean(np.asarray(values) <= 0.0),
        doi_score=lambda xi: np.abs(gayton_limit_state(xi)),
        input_contract={"latent_distribution": "independent standard normal",
                        "physical_transform": "X1=Z1; X2=Z2+3",
                        "failure_definition": "g(X)<=0", "paper_distribution": False,
                        "purpose": "local_failure_domain_software_path"},
        reference_metric=lambda ref: {"name": "failure_probability",
                                      "value": float(np.mean(gayton_limit_state(ref) <= 0.0))},
    )
