"""Make the agents package importable when running pytest from repo root."""

import sys
from pathlib import Path

AGENTS_ROOT = Path(__file__).resolve().parents[1]
if str(AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTS_ROOT))
