import hashlib

import numpy as np

from uqra.adaptive.history import (archived_array_identity,
                                   historical_literal_index_diagnostic)


def test_archived_array_identity_records_shape_dtype_and_sha256(tmp_path):
    path = tmp_path / "pool.npy"
    data = np.arange(24, dtype=np.float64).reshape(2, 12)
    np.save(path, data)
    identity = archived_array_identity(path)
    assert identity["status"] == "recovered"
    assert identity["shape"] == [2, 12]
    assert identity["dtype"] == "float64"
    assert identity["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_literal_history_diagnostic_quantifies_both_index_defects():
    rng = np.random.default_rng(17)
    pool = rng.normal(size=(2, 500))
    first = historical_literal_index_diagnostic(
        pool, seed=9, sample_size=120, retained_count=20,
        doi_size=30, doi_selected_count=12,
    )
    second = historical_literal_index_diagnostic(
        pool, seed=9, sample_size=120, retained_count=20,
        doi_size=30, doi_selected_count=12,
    )
    assert first == second
    assert first["idx01"]["coordinate_identity_mismatches"] == 20
    assert first["idx02"]["global_identity_mismatches"] == 12
    assert first["modern_stable_id_result"] == {
        "cross_order_coordinate_identity_mismatches": 0,
        "doi_global_identity_mismatches": 0,
    }
