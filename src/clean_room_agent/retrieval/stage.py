"""Stage protocol, context, and registry."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from clean_room_agent.llm.client import LLMClient
    from clean_room_agent.query.api import KnowledgeBase
    from clean_room_agent.retrieval.dataclasses import ClassifiedSymbol, ScopedFile, TaskQuery


@dataclass
class StageContext:
    """Mutable context threaded through retrieval stages."""
    task: TaskQuery
    repo_id: int
    repo_path: str
    scoped_files: list[ScopedFile] = field(default_factory=list)
    included_file_ids: set[int] = field(default_factory=set)
    classified_symbols: list[ClassifiedSymbol] = field(default_factory=list)
    tokens_used: int = 0
    stage_timings: dict[str, int] = field(default_factory=dict)
    # Ephemeral per-run config; NOT serialized to session DB.
    retrieval_params: dict = field(default_factory=dict)

    def get_relevant_file_ids(self) -> set[int]:
        """Return file IDs of files marked relevant (not excluded)."""
        return {
            sf.file_id for sf in self.scoped_files
            if sf.relevance == "relevant"
        }

    def to_dict(self) -> dict:
        """Serialize to dict for session DB storage."""
        return {
            "repo_id": self.repo_id,
            "repo_path": self.repo_path,
            "scoped_files": [
                {
                    "file_id": sf.file_id,
                    "path": sf.path,
                    "language": sf.language,
                    "tier": sf.tier,
                    "relevance": sf.relevance,
                    "reason": sf.reason,
                }
                for sf in self.scoped_files
            ],
            "included_file_ids": sorted(self.included_file_ids),
            "classified_symbols": [
                {
                    "symbol_id": cs.symbol_id,
                    "file_id": cs.file_id,
                    "name": cs.name,
                    "kind": cs.kind,
                    "start_line": cs.start_line,
                    "end_line": cs.end_line,
                    "detail_level": cs.detail_level,
                    "reason": cs.reason,
                    "signature": cs.signature,
                    "group_id": cs.group_id,
                }
                for cs in self.classified_symbols
            ],
            "tokens_used": self.tokens_used,
            "stage_timings": self.stage_timings,
        }

    @classmethod
    def from_dict(cls, data: dict, task: TaskQuery) -> StageContext:
        """Deserialize from dict (session DB recovery).

        Required keys: repo_id, repo_path, scoped_files, included_file_ids,
        classified_symbols, tokens_used, stage_timings.
        Missing required keys raise KeyError — corrupt data must not silently
        produce an incomplete context.
        """
        from clean_room_agent.retrieval.dataclasses import ClassifiedSymbol, ScopedFile

        ctx = cls(
            task=task,
            repo_id=data["repo_id"],
            repo_path=data["repo_path"],
        )
        ctx.scoped_files = [
            ScopedFile(**sf) for sf in data["scoped_files"]
        ]
        ctx.included_file_ids = set(data["included_file_ids"])
        ctx.classified_symbols = [
            ClassifiedSymbol(**cs) for cs in data["classified_symbols"]
        ]
        ctx.tokens_used = data["tokens_used"]
        ctx.stage_timings = data["stage_timings"]
        return ctx

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())


@runtime_checkable
class RetrievalStage(Protocol):
    """Protocol for retrieval pipeline stages."""

    @property
    def name(self) -> str: ...

    @property
    def preferred_role(self) -> str:
        """Model role this stage prefers for its LLM calls.

        Every stage must explicitly declare its role — no default.
        Common values: "reasoning", "classifier".
        """
        ...

    def run(
        self,
        context: StageContext,
        kb: KnowledgeBase,
        task: TaskQuery,
        llm: LLMClient,
    ) -> StageContext: ...


@dataclass
class StageInfo:
    """Registry entry for a retrieval stage."""
    cls: type
    description: str


# Module-level stage registry
_STAGE_REGISTRY: dict[str, StageInfo] = {}


def register_stage(name: str, *, description: str):
    """Decorator to register a stage class by name with a description."""
    def decorator(cls):
        if not description:
            raise ValueError(f"Stage {name!r} must have a non-empty description")
        _STAGE_REGISTRY[name] = StageInfo(cls=cls, description=description)
        return cls
    return decorator


def get_stage(name: str) -> RetrievalStage:
    """Instantiate a registered stage by name. Hard error on unknown."""
    if name not in _STAGE_REGISTRY:
        raise ValueError(
            f"Unknown stage {name!r}. Registered: {sorted(_STAGE_REGISTRY.keys())}"
        )
    return _STAGE_REGISTRY[name].cls()


def get_stage_descriptions() -> dict[str, str]:
    """Return {name: description} for all registered stages."""
    return {name: info.description for name, info in _STAGE_REGISTRY.items()}


# Documented defaults for all retrieval parameters (Optional classification).
# These are tuning knobs, not behavioral switches. Override via [retrieval] config.
_RETRIEVAL_DEFAULTS: dict[str, int | float] = {
    "max_deps": 30,
    "max_co_changes": 20,
    "max_metadata": 20,
    "max_keywords": 5,
    "max_symbol_matches": 10,
    "max_callees": 5,
    "max_callers": 5,
    "max_candidate_pairs": 50,
    "min_composite_score": 0.3,
    "max_group_size": 8,
}


def resolve_retrieval_param(params: dict, key: str) -> int | float:
    """Resolve retrieval parameter with documented default. Optional classification."""
    if key not in _RETRIEVAL_DEFAULTS:
        raise KeyError(f"Unknown retrieval parameter: {key!r}")
    value = params.get(key, _RETRIEVAL_DEFAULTS[key])
    if not isinstance(value, (int, float)):
        raise TypeError(f"Retrieval param {key!r} must be numeric, got {type(value).__name__}")
    return value
