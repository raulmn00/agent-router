"""Intent classes used by the router.

Single source of truth — every other module imports INTENTS / LABEL2ID / ID2LABEL
from here so a new class only needs to be added in one place.
"""

from enum import Enum
from typing import Literal

IntentLiteral = Literal["simple_qa", "complex_task", "document_qa", "chitchat"]


class Intent(str, Enum):
    SIMPLE_QA = "simple_qa"
    COMPLEX_TASK = "complex_task"
    DOCUMENT_QA = "document_qa"
    CHITCHAT = "chitchat"


INTENTS: list[str] = [i.value for i in Intent]
LABEL2ID: dict[str, int] = {name: idx for idx, name in enumerate(INTENTS)}
ID2LABEL: dict[int, str] = {idx: name for name, idx in LABEL2ID.items()}
