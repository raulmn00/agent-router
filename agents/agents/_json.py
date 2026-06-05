"""Tolerant JSON extraction for LLM outputs.

LLMs love to wrap JSON in ```json ... ``` fences, prepend "Sure! Here:", or
emit extra prose around the payload. This helper grabs the first plausible
JSON object/array out of arbitrary text.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Return the first JSON value parsed out of `text`.

    Strategy:
      1. If text is itself valid JSON, return it.
      2. Else look for fenced ```json``` blocks and try each.
      3. Else slice from the first `{` or `[` to the matching close and try.
      4. Else raise ValueError.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty LLM output")

    # 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2: fenced blocks
    for match in _FENCE_RE.finditer(text):
        chunk = match.group(1).strip()
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue

    # 3: scan for the first JSON value via bracket matching
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        break  # try the other bracket type
    raise ValueError(f"could not parse JSON from: {text[:200]!r}")
