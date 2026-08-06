# Verified dependency environments

The compatibility suite is locked for two CPython environments:

- `compatibility-py311.txt`: installable but not continuously validated, Python 3.11;
- `compatibility-py312.txt`: sole formal validation baseline, Windows and Python 3.12.

Ubuntu jobs, if retained for aggregation or lightweight checks, are not Ubuntu software
compatibility validation. Python 3.11 is not an M3 completion gate.

Create an isolated environment, install the matching lock, install UQRA without
re-resolving dependencies, and run the gate:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements/compatibility-py311.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest tests/compatibility -q
```

On Windows use `.venv\Scripts\python.exe`. `pyDOE2` is intentionally absent:
UQRA now owns the LHS implementation it uses, avoiding `pyDOE2`'s Python 3.12
failure caused by the removed `imp` module. Runtime dependency ranges remain in
`setup.py`; these lock files define the versions certified by CI.
