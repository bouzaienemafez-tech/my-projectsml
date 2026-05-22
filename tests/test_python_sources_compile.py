from pathlib import Path
import py_compile


def test_python_sources_compile():
    repo_root = Path(__file__).resolve().parents[1]
    excluded_parts = {".git", "__pycache__"}

    python_files = [
        path
        for path in repo_root.rglob("*.py")
        if not excluded_parts.intersection(path.parts)
    ]

    assert python_files, "Expected at least one Python source file."

    for path in python_files:
        py_compile.compile(path, doraise=True)
