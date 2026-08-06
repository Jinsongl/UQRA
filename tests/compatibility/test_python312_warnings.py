from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_uqra_sources_compile_without_python312_warnings():
    process = subprocess.run(
        [
            sys.executable,
            "-W", "error::SyntaxWarning",
            "-W", "error::DeprecationWarning",
            "-m", "compileall",
            "-f", "-q", str(ROOT / "uqra"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
