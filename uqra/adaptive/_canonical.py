"""Private canonical JSON hashing shared by adaptive benchmark manifests."""
from __future__ import annotations

import hashlib
import json


def canonical_json_hash(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
