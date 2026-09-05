"""
conftest.py

Ensures the project root is on sys.path during test collection, regardless
of how pytest is invoked (`pytest`, `python -m pytest`), which OS, or
whether every package under app/ has an __init__.py.

Why this belongs here rather than relying on `python -m pytest`: pytest
always imports a conftest.py it discovers and adds that file's own
directory to sys.path when it does — this is guaranteed pytest behavior,
independent of the import-mode edge cases that caused the original
'ModuleNotFoundError: No module named app' (plain `pytest tests/` inserts
the tests/ directory itself onto sys.path when tests/ lacks an
__init__.py, not the project root).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))