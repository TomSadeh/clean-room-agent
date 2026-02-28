"""Phase 3 execute-stage dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any


# -- Serialization mixin (T81) --


_SERIALIZABLE_PRIMITIVES = (str, int, float, bool, type(None))


class _SerializableMixin:
    """Auto-generate to_dict()/from_dict() for dataclasses.

    Subclass class variables:
        _REQUIRED: field names that must be present in from_dict data
        _NESTED: {field_name: element_type} for list fields of nested dataclasses
        _VALIDATE_LISTS: field names that must be validated as list type (non-nested)
        _EXCLUDE: field names to omit from to_dict output
        _NON_EMPTY: field names that must be truthy (non-empty) after construction

    # INVARIANT: Fields in _REQUIRED must NOT have default_factory values.
    # default_factory would silently supply values, bypassing required-field validation.
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

    @staticmethod
    def _validate_serializable(val: object, field_name: str) -> None:
        """Validate that a value is JSON-serializable (primitive or dict)."""
        if not isinstance(val, (*_SERIALIZABLE_PRIMITIVES, dict)):
            raise TypeError(
                f"Field '{field_name}' has non-serializable type "
                f"{type(val).__name__}. Expected primitive, dict, or "
                f"object with to_dict()."
            )

    def to_dict(self) -> dict:
        result = {}
        for f in dataclass_fields(self):
            if f.name in self._EXCLUDE:
                continue
            val = getattr(self, f.name)
            if isinstance(val, list):
                items = []
                for item in val:
                    if hasattr(item, "to_dict"):
                        items.append(item.to_dict())
                    else:
                        self._validate_serializable(item, f.name)
                        items.append(item)
                result[f.name] = items
            elif hasattr(val, "to_dict"):
                result[f.name] = val.to_dict()
            else:
                self._validate_serializable(val, f.name)
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


# -- Decomposed planning data structures --


@dataclass
class ChangePoint(_SerializableMixin):
    """A single file/symbol that needs to change, identified during enumeration."""
    file_path: str
    symbol: str
    change_type: str  # "modify", "add", "delete"
    rationale: str

    _REQUIRED = ("file_path", "symbol", "change_type", "rationale")
    _NON_EMPTY = ("file_path", "symbol", "change_type", "rationale")


@dataclass
class ChangePointEnumeration(_SerializableMixin):
    """Result of the change point enumeration stage."""
    task_summary: str
    change_points: list[ChangePoint]

    _REQUIRED = ("task_summary", "change_points")
    _NESTED = {"change_points": ChangePoint}
    _NON_EMPTY = ("task_summary", "change_points")


@dataclass
class PartGroup(_SerializableMixin):
    """A logical grouping of change points into a part."""
    id: str
    description: str
    change_point_indices: list[int]
    affected_files: list[str] = field(default_factory=list)

    _REQUIRED = ("id", "description", "change_point_indices")
    _VALIDATE_LISTS = ("change_point_indices", "affected_files")
    _NON_EMPTY = ("id", "description", "change_point_indices")


@dataclass
class PartGrouping(_SerializableMixin):
    """Result of the part grouping stage."""
    parts: list[PartGroup]

    _REQUIRED = ("parts",)
    _NESTED = {"parts": PartGroup}
    _NON_EMPTY = ("parts",)


@dataclass
class SymbolTarget(_SerializableMixin):
    """A specific symbol targeted for modification within a part."""
    file_path: str
    symbol: str
    action: str  # "modify", "add", "delete"
    rationale: str

    _REQUIRED = ("file_path", "symbol", "action", "rationale")
    _NON_EMPTY = ("file_path", "symbol", "action", "rationale")


@dataclass
class SymbolTargetEnumeration(_SerializableMixin):
    """Result of the symbol targeting stage for a single part."""
    part_id: str
    targets: list[SymbolTarget]

    _REQUIRED = ("part_id", "targets")
    _NESTED = {"targets": SymbolTarget}
    _NON_EMPTY = ("part_id", "targets")


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


# -- Decomposed adjustment data structures --


@dataclass
class FailureSignal(_SerializableMixin):
    """A single categorized failure extracted from a StepResult."""
    category: str     # "compile_error", "test_failure", "patch_failure", "unknown"
    message: str      # the raw failure text (budget-truncated if necessary)
    source: str       # "error_info" or "step_failed"

    _REQUIRED = ("category", "message", "source")
    _NON_EMPTY = ("category", "message", "source")


@dataclass
class AdjustmentVerdicts(_SerializableMixin):
    """Aggregated binary verdicts from the decomposed adjustment sub-tasks."""
    step_viability: dict[str, bool]         # step_id -> still valid?
    root_causes: dict[str, list[int]]       # step_id -> [failure_indices it caused]
    new_steps_needed: list[int]             # failure_indices that need new steps
    failure_signals: list[FailureSignal]    # the extracted failures (for context)

    _REQUIRED = ("step_viability", "root_causes", "new_steps_needed", "failure_signals")
    _NESTED = {"failure_signals": FailureSignal}


# -- Decomposed scaffold data structures --


@dataclass
class InterfaceType(_SerializableMixin):
    """A type declaration identified during interface enumeration."""
    name: str           # e.g. "HashEntry"
    kind: str           # "struct" | "enum" | "typedef" | "union"
    fields_description: str
    file_path: str      # which .h file

    _REQUIRED = ("name", "kind", "fields_description", "file_path")
    _NON_EMPTY = ("name", "kind", "file_path")


@dataclass
class InterfaceFunction(_SerializableMixin):
    """A function signature identified during interface enumeration."""
    name: str
    return_type: str
    params: str         # C parameter list text
    purpose: str        # becomes the docstring
    file_path: str      # which .h declares this
    source_file: str    # which .c implements this

    _REQUIRED = ("name", "return_type", "params", "purpose", "file_path", "source_file")
    _NON_EMPTY = ("name", "return_type", "file_path", "source_file")


@dataclass
class InterfaceEnumeration(_SerializableMixin):
    """Result of the interface enumeration stage of decomposed scaffold."""
    types: list[InterfaceType]
    functions: list[InterfaceFunction]
    includes: list[str]  # system headers needed (e.g. "stdlib.h")

    _REQUIRED = ("types", "functions", "includes")
    _NESTED = {"types": InterfaceType, "functions": InterfaceFunction}
    _VALIDATE_LISTS = ("includes",)


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


# -- Compiler error classification --


@dataclass
class CompilerErrorClassification(_SerializableMixin):
    """Result of compiler error classification on the retry path."""
    category: str          # "missing_include", "signature_mismatch", "missing_definition", "logic_error"
    raw_error: str         # the original compiler error string
    suggested_include: str | None = None   # header to add (only for missing_include)
    diagnostic_context: str | None = None  # enriched context for the retry prompt

    _REQUIRED = ("category", "raw_error")
    _NON_EMPTY = ("category", "raw_error")


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
        if self.artifact is not None:
            if not hasattr(self.artifact, "to_dict"):
                raise TypeError(
                    f"PassResult.artifact must implement to_dict(), "
                    f"got {type(self.artifact).__name__}"
                )
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
