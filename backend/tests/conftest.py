"""Make backend imports + sibling packages available when running pytest from repo root."""

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent

# `backend/app` itself
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Sibling packages (router/, agents/, eval/) — same trick the other suites use,
# avoids needing `pip install -e ../router ../agents ../eval` to run the tests.
for sibling in ("router", "agents", "eval"):
    p = REPO_ROOT / sibling
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Give each test a fresh per-IP budget so they don't accumulate 429s."""
    from app.api import limiter

    limiter.reset()
    yield
    limiter.reset()
