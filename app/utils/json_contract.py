"""Robust JSON parser for LLM outputs with fallback handling."""
from __future__ import annotations

import json
import re
from typing import Any


def parse_json_contract(raw: str) -> dict[str, Any]:
    """Parse JSON from potentially messy LLM output.

    Handles:
    - Clean JSON strings
    - Fenced markdown code blocks (```json ... ```)
    - Partial JSON extraction via regex
    - Returns deterministic error shape on total failure
    """
    if not raw or not raw.strip():
        return {"error": "empty_input", "raw": ""}

    cleaned = raw.strip()

    # Strip markdown fenced code blocks: ```json ... ``` or ``` ... ```
    fenced_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    fenced_match = re.search(fenced_pattern, cleaned, re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    # Attempt direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        return {"data": result}
    except json.JSONDecodeError:
        pass

    # Attempt to extract first JSON object via brace matching
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Total failure — return deterministic error shape
    return {"error": "parse_failed", "raw": raw[:500]}
