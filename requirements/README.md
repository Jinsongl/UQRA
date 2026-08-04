# Verified dependency environments

The compatibility suite is locked for two CPython environments:

- `compatibility-py311.txt`: maintenance baseline, Python 3.11;
- `compatibility-py312.txt`: publication target, Python 3.12.

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
