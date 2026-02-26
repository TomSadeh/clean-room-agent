"""Retrieval pipeline dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_TASK_TYPES = ("bug_fix", "feature", "refactor", "test", "docs", "unknown")
VALID_DETAIL_LEVELS = ("primary", "supporting", "type_context", "excluded")


@dataclass
class BudgetConfig:
    """Token budget configuration for a retrieval run."""
    context_window: int
    reserved_tokens: int

    def __post_init__(self):
        if self.context_window <= 0:
            raise ValueError(f"context_window must be > 0, got {self.context_window}")
        if self.reserved_tokens < 0:
            raise ValueError(f"reserved_tokens must be >= 0, got {self.reserved_tokens}")
        if self.reserved_tokens >= self.context_window:
            raise ValueError(
                f"reserved_tokens ({self.reserved_tokens}) must be < "
                f"context_window ({self.context_window})"
            )

    @property
    def effective_budget(self) -> int:
        """Tokens available for retrieval context."""
        return self.context_window - self.reserved_tokens


@dataclass
class TaskQuery:
    """Parsed and enriched task description."""
    raw_task: str
    task_id: str
    mode: str
    repo_id: int
    mentioned_files: list[str] = field(default_factory=list)
    mentioned_symbols: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    error_patterns: list[str] = field(default_factory=list)
    task_type: str = "unknown"
    intent_summary: str = ""
    seed_file_ids: list[int] = field(default_factory=list)
    seed_symbol_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if self.task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {VALID_TASK_TYPES}, got {self.task_type!r}"
            )


VALID_RELEVANCE = ("pending", "relevant", "irrelevant")


@dataclass
class ScopedFile:
    """A file included in scope with tier and relevance metadata."""
    file_id: int
    path: str
    language: str
    tier: int  # 0=plan, 1=seed, 2=dep, 3=co-change, 4=metadata
    relevance: str = "pending"  # pending, relevant, irrelevant
    reason: str = ""

    def __post_init__(self):
        if self.relevance not in VALID_RELEVANCE:
            raise ValueError(
                f"relevance must be one of {VALID_RELEVANCE}, got {self.relevance!r}"
            )


@dataclass
class ClassifiedSymbol:
    """A symbol classified by the precision stage."""
    symbol_id: int
    file_id: int
    name: str
    kind: str
    start_line: int
    end_line: int
    detail_level: str = "excluded"  # primary, supporting, type_context, excluded
    reason: str = ""
    signature: str = ""
    group_id: str | None = None
    file_source: str = "project"

    def __post_init__(self):
        if self.detail_level not in VALID_DETAIL_LEVELS:
            raise ValueError(
                f"detail_level must be one of {VALID_DETAIL_LEVELS}, got {self.detail_level!r}"
            )


VALID_FILE_DETAIL_LEVELS = ("primary", "supporting", "type_context")


@dataclass
class FileContent:
    """A rendered file entry in the final context package."""
    file_id: int
    path: str
    language: str
    content: str
    token_estimate: int
    detail_level: str  # primary, supporting, type_context
    included_symbols: list[str] = field(default_factory=list)
    metadata_summary: str = ""

    def __post_init__(self):
        if self.detail_level not in VALID_FILE_DETAIL_LEVELS:
            raise ValueError(
                f"detail_level must be one of {VALID_FILE_DETAIL_LEVELS}, got {self.detail_level!r}"
            )


@dataclass
class ContextPackage:
    """The final curated context delivered to the execute stage."""
    task: TaskQuery
    files: list[FileContent] = field(default_factory=list)
    total_token_estimate: int = 0
    budget: BudgetConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    environment_brief: str = ""

    def to_prompt_text(self) -> str:
        """Render the context package as prompt text for the execute stage.

        R5: Uses XML-style <code> tags instead of triple backticks to prevent
        content injection (code containing ``` would break markdown fences).
        """
        parts = []
        if self.environment_brief:
            parts.append(self.environment_brief)
        parts.append(f"# Task\n{self.task.raw_task}\n")
        if self.task.intent_summary:
            parts.append(f"# Intent\n{self.task.intent_summary}\n")
        for fc in self.files:
            header = f"## {fc.path} [{fc.language}] ({fc.detail_level})"
            meta_line = f"{fc.metadata_summary}\n" if fc.metadata_summary else ""
            parts.append(f"{header}\n{meta_line}<code lang=\"{fc.language}\">\n{fc.content}\n</code>\n")
        return "\n".join(parts)


@dataclass
class RefinementRequest:
    """Request for pipeline re-entry with additional context.

    The orchestrator creates a new task_id for each refinement pass.
    ``source_task_id`` identifies the previous pipeline run whose session
    archive should be restored for re-entry.
    """
    reason: str
    source_task_id: str
    missing_files: list[str] = field(default_factory=list)
    missing_symbols: list[str] = field(default_factory=list)
    missing_tests: list[str] = field(default_factory=list)
    error_patterns: list[str] = field(default_factory=list)
