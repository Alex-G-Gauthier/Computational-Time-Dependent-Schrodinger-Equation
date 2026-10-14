"""Shared pytest configuration for the tdse test suite.

The single non-trivial thing this file does is ensure the ``tdse``
package is importable when the suite is run *without* having first done
``pip install -e .``.  This keeps the tests runnable in three settings:

1. After ``pip install -e .`` from the repo root (CI does this).
2. Via ``pytest`` invoked from the repo root with no install.
3. Via ``python -m pytest tests`` from the repo root.

We also silence the RuntimeWarnings deliberately raised by the FTCS
overflow demonstration so test output stays clean.
"""

from __future__ import annotations

import os
import sys
import warnings


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


warnings.filterwarnings("ignore", category=RuntimeWarning)
