"""Reading a model's generated answer."""
from __future__ import annotations

import re

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_letter(text, valid) -> str:
    """First answer letter in a generated answer that is one of `valid`; '' when there is none."""
    s = str(text).strip().upper()
    if s[:1] in valid and (len(s) == 1 or not s[1].isalnum()):
        return s[0]
    m = re.search(r"\b([A-Z])\b", s)
    return m.group(1) if m and m.group(1) in valid else ""


def parse_number(text) -> float:
    """First number in a generated answer; 0.0 when there is none."""
    m = NUMBER.search(str(text))
    return float(m.group(0)) if m else 0.0
