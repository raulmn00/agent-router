import sys
from pathlib import Path

from dotenv import load_dotenv

# Make sibling packages (router, agents, eval) importable when running uvicorn
# without `pip install -e ../<sibling>` for each — the Dockerfile bakes the
# same paths into PYTHONPATH for production.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sib in ("router", "agents", "eval"):
    _p = _REPO_ROOT / _sib
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

load_dotenv()
