"""Make the eval package + sibling packages importable for pytest."""

import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVAL_ROOT.parent

if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

for sibling in ("router", "agents"):
    p = REPO_ROOT / sibling
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
