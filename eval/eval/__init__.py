import sys
from pathlib import Path

from dotenv import load_dotenv

# Make sibling packages (router/, agents/) importable when running `python -m eval.*`
# from this repo without `pip install -e ../router ../agents`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sib in ("router", "agents"):
    _p = _REPO_ROOT / _sib
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

load_dotenv()
