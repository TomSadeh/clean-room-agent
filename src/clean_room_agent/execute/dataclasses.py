"""Phase 3 execute-stage dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any


# -- Serialization mixin (T81) --


class _SerializableMixin:
    """Auto-generate to_dict()/from_dict() for dataclasses.

    Subclass class variables:
        _REQUIRED: field names that must be present in from_dict data
        _NESTED: {field_name: element_type} for list fields of nested dataclasses
        _VALIDATE_LISTS: field names that must be validated as list type (non-nested)
        _EXCLUDE: field names to omit from to_dict output
        _NON_EMPTY: field names that must be truthy (non-empty) after construction
    """

    _REQUIRED: tuple[str, ...] = ()
    _NESTED: dict[str, type] = {}
    _VALIDATE_LISTS: tuple[str, ...] = ()
    _EXCLUDE: frozenset[str] = frozenset()
    _NON_EMPTY: tuple[str, ...] = ()

    def __post_init__(self):
        for name in self._NON_EMPTY:
            if not getattr(self, name):
                raise ValueError(f"{type(self).__name__}.{name} must be non-empty")

    def to_dict(self) -> dict:
        result = {}
        for f in dataclass_fields(self):
            if f.name in self._EXCLUDE:
                continue
            val = getattr(self, f.name)
            if isinstance(val, list):
                result[f.name] = [
                    item.to_dict() if hasattr(item, "to_dict") else item
                    for item in val
                ]
            elif hasattr(val, "to_dict"):
                result[f.name] = val.to_dict()
            else:
                result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, data: dict):
        # Validate required keys
        for key in cls._REQUIRED:
            if key not in data:
                raise ValueError(f"{cls.__name__}.from_dict missing required key: {key!r}")

        # Validate list fields (both nested and plain)
        for key in set(cls._NESTED) | set(cls._VALIDATE_LISTS):
            if key in data and not isinstance(data[key], list):
                raise ValueError(
                    f"{cls.__name__}.from_dict: '{key}' must be a list, "
                    f"got {type(data[key]).__name__}"
                )

        # Build kwargs — only include fields present in data; let dataclass
        # defaults handle anything missing (optional fields).
        kwargs = {}
        for f in dataclass_fields(cls):
            if f.name not in data:
                continue
            val = data[f.name]
            if f.name in cls._NESTED:
                elem_type = cls._NESTED[f.name]
                kwargs[f.name] = [elem_type.from_dict(x) for x in val]
            else:
                kwargs[f.name] = val

        return cls(**kwargs)


# -- Plan data structures (orchestrator-internal) --


@dataclass
class PlanStep(_SerializableMixin):
    """A single implementation step within a part plan."""
    id: str
    description: str
    target_files: list[str] = field(default_factory=list)
    target_symbols: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    _REQUIRED = ("id", "description")
    _NON_EMPTY = ("id", "description")


@dataclass
class MetaPlanPart(_SerializableMixin):
    """A high-level part in the meta-plan decomposition."""
    id: str
    description: str
    affected_files: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    _REQUIRED = ("id", "description")
    _NON_EMPTY = ("id", "description")


@dataclass
class MetaPlan(_SerializableMixin):
    """Top-level task decomposition produced by the meta-plan pass."""
    task_summary: str
    parts: list[MetaPlanPart]
    rationale: str

    _REQUIRED = ("task_summary", "parts", "rationale")
    _NESTED = {"parts": MetaPlanPart}
    _NON_EMPTY = ("task_summary", "parts", "rationale")


@dataclass
class PartPlan(_SerializableMixin):
    """Detailed step plan for a single part."""
    part_id: str
    task_summary: str
    steps: list[PlanStep]
    rationale: str

    _REQUIRED = ("part_id", "task_summary", "steps", "rationale")
    _NESTED = {"steps": PlanStep}
    _NON_EMPTY = ("part_id", "task_summary", "steps", "rationale")


@dataclass
class PlanAdjustment(_SerializableMixin):
    """Revised plan steps after an adjustment pass."""
    revised_steps: list[PlanStep]
    rationale: str
    changes_made: list[str]

    _REQUIRED = ("revised_steps", "rationale", "changes_made")
    _NESTED = {"revised_steps": PlanStep}
    _VALIDATE_LISTS = ("changes_made",)
    _NON_EMPTY = ("rationale",)


# -- User-facing plan format --

@dataclass
class PlanArtifact(_SerializableMixin):
    """User-facing plan output (per pipeline-and-modes.md Section 3.3)."""
    task_summary: str
    affected_files: list[dict[str, Any]]
    execution_order: list[str]
    rationale: str

    _REQUIRED = ("task_summary", "affected_files", "execution_order", "rationale")
    _NON_EMPTY = ("task_summary",)

    @classmethod
    def from_meta_plan(cls, meta_plan: MetaPlan) -> PlanArtifact:
        """Convert internal MetaPlan to user-facing PlanArtifact."""
        affected_files = []
        seen_files: set[str] = set()
        for part in meta_plan.parts:
            for f in part.affected_files:
                if f not in seen_files:
                    seen_files.add(f)
                    affected_files.append({"path": f, "role": "modified", "changes": part.description})

        execution_order = [p.id for p in meta_plan.parts]

        return cls(
            task_summary=meta_plan.task_summary,
            affected_files=affected_files,
            execution_order=execution_order,
            rationale=meta_plan.rationale,
        )


# -- Implementation data structures --

@dataclass
class PatchEdit(_SerializableMixin):
    """A single search/replace edit."""
    file_path: str
    search: str
    replacement: str

    _REQUIRED = ("file_path", "search", "replacement")
    _NON_EMPTY = ("file_path", "search")


@dataclass
class StepResult(_SerializableMixin):
    """Result of a single implementation step."""
    success: bool
    edits: list[PatchEdit] = field(default_factory=list)
    error_info: str | None = None
    raw_response: str = ""

    _REQUIRED = ("success",)
    _NESTED = {"edits": PatchEdit}


@dataclass
class PatchResult(_SerializableMixin):
    """Result of applying edits to the filesystem."""
    success: bool
    files_modified: list[str] = field(default_factory=list)
    error_info: str | None = None
    original_contents: dict[str, str] = field(default_factory=dict)

    _REQUIRED = ("success",)
    _EXCLUDE = frozenset({"original_contents"})


# -- Scaffold data structures --

@dataclass
class FunctionStub(_SerializableMixin):
    """A function stub from the scaffold, to be implemented independently."""
    name: str
    file_path: str
    signature: str
    docstring: str
    start_line: int
    end_line: int
    dependencies: list[str] = field(default_factory=list)
    header_file: str | None = None

    _REQUIRED = ("name", "file_path", "signature", "start_line", "end_line")
    _NON_EMPTY = ("name", "file_path", "signature")


@dataclass
class ScaffoldResult(_SerializableMixin):
    """Result of scaffold generation."""
    success: bool
    edits: list[PatchEdit] = field(default_factory=list)
    header_files: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    function_stubs: list[FunctionStub] = field(default_factory=list)
    error_info: str | None = None
    raw_response: str = ""
    compilation_output: str | None = None

    _REQUIRED = ("success",)
    _NESTED = {"edits": PatchEdit, "function_stubs": FunctionStub}


# -- Validation data structures --

@dataclass
class ValidationResult(_SerializableMixin):
    """Result of test/lint/type-check validation."""
    success: bool
    test_output: str | None = None
    lint_output: str | None = None
    type_check_output: str | None = None
    failing_tests: list[str] = field(default_factory=list)

    _REQUIRED = ("success",)


# -- Orchestrator data structures --

@dataclass
class PassResult(_SerializableMixin):
    """Result of a single orchestrator pass."""
    pass_type: str
    success: bool
    task_run_id: int | None = None
    artifact: Any = None

    _REQUIRED = ("pass_type", "success")
    _NON_EMPTY = ("pass_type",)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "pass_type": self.pass_type,
            "task_run_id": self.task_run_id,
            "success": self.success,
        }
        if self.artifact is not None and hasattr(self.artifact, "to_dict"):
            d["artifact"] = self.artifact.to_dict()
        return d


VALID_ORCHESTRATOR_STATUSES = ("complete", "partial", "failed")


@dataclass
class OrchestratorResult(_SerializableMixin):
    """Final result of a full orchestrator run."""
    task_id: str
    status: str
    parts_completed: int = 0
    steps_completed: int = 0
    cumulative_diff: str = ""
    pass_results: list[PassResult] = field(default_factory=list)

    _REQUIRED = ("task_id", "status")
    _NESTED = {"pass_results": PassResult}
    _NON_EMPTY = ("task_id",)

    def __post_init__(self):
        super().__post_init__()
        if self.status not in VALID_ORCHESTRATOR_STATUSES:
            raise ValueError(
                f"OrchestratorResult.status must be one of "
                f"{VALID_ORCHESTRATOR_STATUSES}, got {self.status!r}"
            )
