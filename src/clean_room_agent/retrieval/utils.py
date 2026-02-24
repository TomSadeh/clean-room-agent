"""Shared retrieval utilities."""

import json


def parse_json_response(text: str, context: str = "LLM") -> list | dict:
    """Parse JSON from LLM response, stripping markdown fencing.

    Args:
        text: Raw LLM response text.
        context: Label for error messages (e.g. "scope judgment", "precision").
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        while lines and lines[-1].strip() in ("```", ""):
            if lines[-1].strip() == "```":
                lines.pop()
                break
            lines.pop()
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse {context} JSON: {e}\nRaw: {text}") from e
