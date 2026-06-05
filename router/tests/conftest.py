"""Pytest config — make the router package importable when running from repo root."""

import sys
from pathlib import Path

ROUTER_ROOT = Path(__file__).resolve().parents[1]
if str(ROUTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ROUTER_ROOT))
