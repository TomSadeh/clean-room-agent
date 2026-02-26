"""Centralized file extension / language constants (A17)."""

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
}

KNOWN_EXTENSIONS: frozenset[str] = frozenset(LANGUAGE_MAP)

# TS/JS resolution extensions for import resolution
TS_JS_RESOLVE_EXTENSIONS: tuple[str, ...] = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


def language_from_extension(file_path: str) -> str | None:
    """Map file path to language name via extension, or None if unsupported."""
    for ext, lang in LANGUAGE_MAP.items():
        if file_path.endswith(ext):
            return lang
    return None
