# src/irene_dynamic_examples.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import json

from .irene_style_examples import StyleExample

# Output from train_irene_from_feedback.py
DYNAMIC_FILE = Path("data/irene_style_from_feedback.json")


def load_dynamic_style_examples() -> Dict[str, List[StyleExample]]:
    """
    Load extra style examples mined from chat feedback.

    File format (per lang):
    {
      "en": [{ "user": "...", "assistant": "..." }, ...],
      "vi": [...]
    }

    We convert them into StyleExample with label=None so they behave
    like general style examples.
    """
    # Default empty structure
    empty: Dict[str, List[StyleExample]] = {"en": [], "vi": []}

    if not DYNAMIC_FILE.exists():
        return empty

    try:
        with DYNAMIC_FILE.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        # If the file is corrupted, just ignore it
        return empty

    out: Dict[str, List[StyleExample]] = {"en": [], "vi": []}

    for lang in ("en", "vi"):
        items = raw.get(lang) or []
        cleaned: List[StyleExample] = []
        for it in items:
            user = (it.get("user") or "").strip()
            assistant = (it.get("assistant") or "").strip()
            if not user or not assistant:
                continue
            cleaned.append(
                StyleExample(
                    label=None,
                    user=user,
                    assistant=assistant,
                )
            )
        out[lang] = cleaned

    return out
