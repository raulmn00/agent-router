"""Make backend imports + sibling packages available when running pytest from repo root."""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent

# `backend/app` itself
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Sibling packages (router/, agents/) — same trick the other suites use,
# avoids needing `pip install -e ../router ../agents` to run the tests.
for sibling in ("router", "agents"):
    p = REPO_ROOT / sibling
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
