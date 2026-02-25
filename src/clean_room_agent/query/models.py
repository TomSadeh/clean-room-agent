"""Dataclasses for query API results."""

from dataclasses import dataclass, field


@dataclass
class File:
    id: int
    repo_id: int
    path: str
    language: str
    content_hash: str
    size_bytes: int
    file_source: str = "project"


@dataclass
class Symbol:
    id: int
    file_id: int
    name: str
    kind: str
    start_line: int
    end_line: int
    signature: str | None = None
    parent_symbol_id: int | None = None


@dataclass
class Docstring:
    id: int
    file_id: int
    content: str
    format: str | None = None
    parsed_fields: str | None = None
    symbol_id: int | None = None


@dataclass
class Comment:
    id: int
    file_id: int
    line: int
    content: str
    kind: str | None = None
    is_rationale: bool = False
    symbol_id: int | None = None


@dataclass
class Dependency:
    id: int
    source_file_id: int
    target_file_id: int
    kind: str


@dataclass
class Commit:
    id: int
    repo_id: int
    hash: str
    author: str | None
    message: str | None
    timestamp: str
    files_changed: int | None = None
    insertions: int | None = None
    deletions: int | None = None


@dataclass
class CoChange:
    file_a_id: int
    file_b_id: int
    count: int
    last_commit_hash: str | None = None


@dataclass
class FileMetadata:
    file_id: int
    purpose: str | None = None
    module: str | None = None
    domain: str | None = None
    concepts: str | None = None
    public_api_surface: str | None = None
    complexity_notes: str | None = None


@dataclass
class AdapterInfo:
    id: int
    stage_name: str
    base_model: str
    model_tag: str
    version: int
    performance_notes: str | None = None
    deployed_at: str = ""


@dataclass
class FileContext:
    file: File
    symbols: list[Symbol] = field(default_factory=list)
    docstrings: list[Docstring] = field(default_factory=list)
    rationale_comments: list[Comment] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    co_changes: list[CoChange] = field(default_factory=list)
    recent_commits: list[Commit] = field(default_factory=list)


@dataclass
class RepoOverview:
    repo_id: int
    file_count: int
    language_counts: dict[str, int] = field(default_factory=dict)
    domain_counts: dict[str, int] = field(default_factory=dict)
    most_connected_files: list[tuple[str, int]] = field(default_factory=list)
