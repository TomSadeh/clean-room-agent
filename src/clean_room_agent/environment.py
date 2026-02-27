"""Environment brief: OS, languages, test framework, coding style.

Injected into every LLM prompt so the model doesn't waste tokens
inferring environmental context from code samples.
"""

import platform
import sys
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from clean_room_agent.config import _DEFAULT_CODING_STYLE
from clean_room_agent.query.api import KnowledgeBase

# Coding style enum -> predefined text.  We control the text, not the user.
# Free text in config that enters every prompt is a prompt injection vector.
CODING_STYLES: dict[str, str] = {
    "development": (
        "Development mode. Optimize for fast debugging, not graceful degradation. "
        "No fallbacks and no hardcoded defaults in core logic. "
        "Keep try/except blocks minimal and intentional. "
        "Prefer fail-fast behavior so incorrect assumptions break hard and early."
    ),
    "maintenance": (
        "Maintenance mode. Optimize for stability and backward compatibility. "
        "Comprehensive error handling with informative messages. "
        "Preserve existing interfaces and add deprecation warnings before removal."
    ),
    "prototyping": (
        "Prototyping mode. Optimize for speed of iteration. "
        "Minimal structure, skip non-essential validation. "
        "Favor quick experiments over production-quality code."
    ),
}


@dataclass
class EnvironmentBrief:
    """Universal environment context injected into every LLM prompt."""

    os_name: str
    languages: dict[str, int] = field(default_factory=dict)
    test_framework: str = ""
    coding_style: str = "development"
    file_count: int = 0
    runtime_version: str | None = None

    def to_prompt_text(self) -> str:
        """Render to prompt-ready XML text (~200 tokens)."""
        lines = ["<environment>"]
        lines.append(f"OS: {self.os_name}")

        if self.languages:
            lang_parts = [f"{lang} ({count} files)" for lang, count in
                          sorted(self.languages.items(), key=lambda x: x[1], reverse=True)]
            lines.append(f"Languages: {', '.join(lang_parts)}")

        if self.test_framework:
            lines.append(f"Test framework: {self.test_framework}")

        if self.runtime_version:
            lines.append(f"Runtime: {self.runtime_version}")

        lines.append(f"Files indexed: {self.file_count}")

        # coding_style is validated at construction; KeyError here means a bug
        style_text = CODING_STYLES[self.coding_style]
        lines.append(f"Coding style: {style_text}")

        lines.append("</environment>")
        return "\n".join(lines)


def build_environment_brief(
    config: dict,
    kb: KnowledgeBase,
    repo_id: int,
) -> EnvironmentBrief:
    """Build environment brief from config + knowledge base. Deterministic, no LLM."""
    overview = kb.get_repo_overview(repo_id)

    # Test framework from config — [testing] is Optional (repos may not have tests)
    testing_config = config["testing"]
    test_command = testing_config["test_command"]
    test_framework = test_command.split()[0] if test_command.strip() else ""

    # Coding style from config — [environment] is Optional, _DEFAULT_CODING_STYLE if absent
    env_config = config["environment"]
    coding_style = env_config["coding_style"]
    if coding_style not in CODING_STYLES:
        raise ValueError(
            f"Unknown coding_style {coding_style!r}. "
            f"Valid options: {', '.join(sorted(CODING_STYLES))}"
        )

    # Runtime version if python is a project language
    runtime_version = None
    if "python" in overview.language_counts:
        runtime_version = f"Python {sys.version.split()[0]}"

    return EnvironmentBrief(
        os_name=platform.system(),
        languages=dict(overview.language_counts),
        test_framework=test_framework,
        coding_style=coding_style,
        file_count=overview.file_count,
        runtime_version=runtime_version,
    )


def build_repo_file_tree(kb: KnowledgeBase, repo_id: int) -> str:
    """Build an indented directory tree from all indexed file paths.

    No cap -- the bird's eye prompt has minimal other context.
    R1: no truncation of data the model needs to see.
    """
    files = kb.get_files(repo_id)
    if not files:
        return "(empty repository)"

    paths = sorted(f.path for f in files)

    lines = []
    for path in paths:
        parts = PurePosixPath(path).parts
        indent = "  " * (len(parts) - 1)
        lines.append(f"{indent}{parts[-1]}")

    return "\n".join(lines)
