"""Phase 3 execute-stage dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# -- Plan data structures (orchestrator-internal) --


@dataclass
class PlanStep:
    """A single implementation step within a part plan."""
    id: str
    description: str
    target_files: list[str] = field(default_factory=list)
    target_symbols: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            raise ValueError("PlanStep.id must be non-empty")
        if not self.description:
            raise ValueError("PlanStep.description must be non-empty")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "target_files": self.target_files,
            "target_symbols": self.target_symbols,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlanStep:
        for key in ("id", "description"):
            if key not in data:
                raise ValueError(f"PlanStep.from_dict missing required key: {key!r}")
        return cls(
            id=data["id"],
            description=data["description"],
            target_files=data.get("target_files", []),
            target_symbols=data.get("target_symbols", []),
            depends_on=data.get("depends_on", []),
        )


@dataclass
class MetaPlanPart:
    """A high-level part in the meta-plan decomposition."""
    id: str
    description: str
    affected_files: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            raise ValueError("MetaPlanPart.id must be non-empty")
        if not self.description:
            raise ValueError("MetaPlanPart.description must be non-empty")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "affected_files": self.affected_files,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MetaPlanPart:
        for key in ("id", "description"):
            if key not in data:
                raise ValueError(f"MetaPlanPart.from_dict missing required key: {key!r}")
        return cls(
            id=data["id"],
            description=data["description"],
            affected_files=data.get("affected_files", []),
            depends_on=data.get("depends_on", []),
        )


@dataclass
class MetaPlan:
    """Top-level task decomposition produced by the meta-plan pass."""
    task_summary: str
    parts: list[MetaPlanPart]
    rationale: str

    def __post_init__(self):
        if not self.task_summary:
            raise ValueError("MetaPlan.task_summary must be non-empty")
        if not self.parts:
            raise ValueError("MetaPlan.parts must be non-empty")
        if not self.rationale:
            raise ValueError("MetaPlan.rationale must be non-empty")

    def to_dict(self) -> dict:
        return {
            "task_summary": self.task_summary,
            "parts": [p.to_dict() for p in self.parts],
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MetaPlan:
        for key in ("task_summary", "parts", "rationale"):
            if key not in data:
                raise ValueError(f"MetaPlan.from_dict missing required key: {key!r}")
        if not isinstance(data["parts"], list):
            raise ValueError(
                f"MetaPlan.from_dict: 'parts' must be a list, got {type(data['parts']).__name__}"
            )
        return cls(
            task_summary=data["task_summary"],
            parts=[MetaPlanPart.from_dict(p) for p in data["parts"]],
            rationale=data["rationale"],
        )


@dataclass
class PartPlan:
    """Detailed step plan for a single part."""
    part_id: str
    task_summary: str
    steps: list[PlanStep]
    rationale: str

    def __post_init__(self):
        if not self.part_id:
            raise ValueError("PartPlan.part_id must be non-empty")
        if not self.task_summary:
            raise ValueError("PartPlan.task_summary must be non-empty")
        if not self.steps:
            raise ValueError("PartPlan.steps must be non-empty")
        if not self.rationale:
            raise ValueError("PartPlan.rationale must be non-empty")

    def to_dict(self) -> dict:
        return {
            "part_id": self.part_id,
            "task_summary": self.task_summary,
            "steps": [s.to_dict() for s in self.steps],
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PartPlan:
        for key in ("part_id", "task_summary", "steps", "rationale"):
            if key not in data:
                raise ValueError(f"PartPlan.from_dict missing required key: {key!r}")
        if not isinstance(data["steps"], list):
            raise ValueError(
                f"PartPlan.from_dict: 'steps' must be a list, got {type(data['steps']).__name__}"
            )
        return cls(
            part_id=data["part_id"],
            task_summary=data["task_summary"],
            steps=[PlanStep.from_dict(s) for s in data["steps"]],
            rationale=data["rationale"],
        )


@dataclass
class PlanAdjustment:
    """Revised plan steps after an adjustment pass."""
    revised_steps: list[PlanStep]
    rationale: str
    changes_made: list[str]

    def __post_init__(self):
        if not self.rationale:
            raise ValueError("PlanAdjustment.rationale must be non-empty")

    def to_dict(self) -> dict:
        return {
            "revised_steps": [s.to_dict() for s in self.revised_steps],
            "rationale": self.rationale,
            "changes_made": self.changes_made,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlanAdjustment:
        for key in ("revised_steps", "rationale", "changes_made"):
            if key not in data:
                raise ValueError(f"PlanAdjustment.from_dict missing required key: {key!r}")
        if not isinstance(data["revised_steps"], list):
            raise ValueError(
                f"PlanAdjustment.from_dict: 'revised_steps' must be a list, "
                f"got {type(data['revised_steps']).__name__}"
            )
        if not isinstance(data["changes_made"], list):
            raise ValueError(
                f"PlanAdjustment.from_dict: 'changes_made' must be a list, "
                f"got {type(data['changes_made']).__name__}"
            )
        return cls(
            revised_steps=[PlanStep.from_dict(s) for s in data["revised_steps"]],
            rationale=data["rationale"],
            changes_made=data["changes_made"],
        )


# -- User-facing plan format --

@dataclass
class PlanArtifact:
    """User-facing plan output (per pipeline-and-modes.md Section 3.3)."""
    task_summary: str
    affected_files: list[dict[str, Any]]
    execution_order: list[str]
    rationale: str

    def __post_init__(self):
        if not self.task_summary:
            raise ValueError("PlanArtifact.task_summary must be non-empty")

    def to_dict(self) -> dict:
        return {
            "task_summary": self.task_summary,
            "affected_files": self.affected_files,
            "execution_order": self.execution_order,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlanArtifact:
        for key in ("task_summary", "affected_files", "execution_order", "rationale"):
            if key not in data:
                raise ValueError(f"PlanArtifact.from_dict missing required key: {key!r}")
        return cls(
            task_summary=data["task_summary"],
            affected_files=data["affected_files"],
            execution_order=data["execution_order"],
            rationale=data["rationale"],
        )

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
class PatchEdit:
    """A single search/replace edit."""
    file_path: str
    search: str
    replacement: str

    def __post_init__(self):
        if not self.file_path:
            raise ValueError("PatchEdit.file_path must be non-empty")
        if not self.search:
            raise ValueError("PatchEdit.search must be non-empty")

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "search": self.search,
            "replacement": self.replacement,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PatchEdit:
        for key in ("file_path", "search", "replacement"):
            if key not in data:
                raise ValueError(f"PatchEdit.from_dict missing required key: {key!r}")
        return cls(
            file_path=data["file_path"],
            search=data["search"],
            replacement=data["replacement"],
        )


@dataclass
class StepResult:
    """Result of a single implementation step."""
    success: bool
    edits: list[PatchEdit] = field(default_factory=list)
    error_info: str | None = None
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "edits": [e.to_dict() for e in self.edits],
            "error_info": self.error_info,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StepResult:
        if "success" not in data:
            raise ValueError("StepResult.from_dict missing required key: 'success'")
        return cls(
            success=data["success"],
            edits=[PatchEdit.from_dict(e) for e in data.get("edits", [])],
            error_info=data.get("error_info"),
            raw_response=data.get("raw_response", ""),
        )


@dataclass
class PatchResult:
    """Result of applying edits to the filesystem."""
    success: bool
    files_modified: list[str] = field(default_factory=list)
    error_info: str | None = None
    original_contents: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "files_modified": self.files_modified,
            "error_info": self.error_info,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PatchResult:
        if "success" not in data:
            raise ValueError("PatchResult.from_dict missing required key: 'success'")
        return cls(
            success=data["success"],
            files_modified=data.get("files_modified", []),
            error_info=data.get("error_info"),
        )


# -- Validation data structures --

@dataclass
class ValidationResult:
    """Result of test/lint/type-check validation."""
    success: bool
    test_output: str | None = None
    lint_output: str | None = None
    type_check_output: str | None = None
    failing_tests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "test_output": self.test_output,
            "lint_output": self.lint_output,
            "type_check_output": self.type_check_output,
            "failing_tests": self.failing_tests,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ValidationResult:
        if "success" not in data:
            raise ValueError("ValidationResult.from_dict missing required key: 'success'")
        return cls(
            success=data["success"],
            test_output=data.get("test_output"),
            lint_output=data.get("lint_output"),
            type_check_output=data.get("type_check_output"),
            failing_tests=data.get("failing_tests", []),
        )


# -- Orchestrator data structures --

@dataclass
class PassResult:
    """Result of a single orchestrator pass."""
    pass_type: str
    success: bool
    task_run_id: int | None = None
    artifact: Any = None

    def __post_init__(self):
        if not self.pass_type:
            raise ValueError("PassResult.pass_type must be non-empty")

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "pass_type": self.pass_type,
            "task_run_id": self.task_run_id,
            "success": self.success,
        }
        if self.artifact is not None and hasattr(self.artifact, "to_dict"):
            d["artifact"] = self.artifact.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PassResult:
        for key in ("pass_type", "success"):
            if key not in data:
                raise ValueError(f"PassResult.from_dict missing required key: {key!r}")
        return cls(
            pass_type=data["pass_type"],
            success=data["success"],
            task_run_id=data.get("task_run_id"),
        )


VALID_ORCHESTRATOR_STATUSES = ("complete", "partial", "failed")


@dataclass
class OrchestratorResult:
    """Final result of a full orchestrator run."""
    task_id: str
    status: str
    parts_completed: int = 0
    steps_completed: int = 0
    cumulative_diff: str = ""
    pass_results: list[PassResult] = field(default_factory=list)

    def __post_init__(self):
        if not self.task_id:
            raise ValueError("OrchestratorResult.task_id must be non-empty")
        if self.status not in VALID_ORCHESTRATOR_STATUSES:
            raise ValueError(
                f"OrchestratorResult.status must be one of "
                f"{VALID_ORCHESTRATOR_STATUSES}, got {self.status!r}"
            )

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "parts_completed": self.parts_completed,
            "steps_completed": self.steps_completed,
            "cumulative_diff": self.cumulative_diff,
            "pass_results": [pr.to_dict() for pr in self.pass_results],
        }

    @classmethod
    def from_dict(cls, data: dict) -> OrchestratorResult:
        for key in ("task_id", "status"):
            if key not in data:
                raise ValueError(f"OrchestratorResult.from_dict missing required key: {key!r}")
        return cls(
            task_id=data["task_id"],
            status=data["status"],
            parts_completed=data.get("parts_completed", 0),
            steps_completed=data.get("steps_completed", 0),
            cumulative_diff=data.get("cumulative_diff", ""),
            pass_results=[
                PassResult.from_dict(pr) for pr in data.get("pass_results", [])
            ],
        )
