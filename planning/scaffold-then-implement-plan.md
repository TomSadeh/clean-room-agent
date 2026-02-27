# Implementation Plan: Scaffold-Then-Implement

Design record: `protocols/design_records/scaffold_then_implement.md`

## Context

The current orchestrator flow is: `meta_plan → part_plan → [step_implement loop] → doc_pass → test_plan → test_implement → validate`. Each implement step receives a full ContextPackage (potentially 20K+ tokens) and produces search/replace edits across arbitrary files. The model sees everything and must reason about function boundaries, type definitions, and cross-file consistency simultaneously.

The design record proposes inserting a **scaffold step** between part_plan and the step_implement loop. The scaffold produces compilable C code with complete headers and empty function stubs. Each function is then implemented by an independent LLM call against the compact scaffold (3-5K tokens instead of 20K+).

This is C-specific — it exploits C's explicit dependency graph (headers declare, source files implement).

## What Changes

### 1. New dataclasses — `ScaffoldResult` in `execute/dataclasses.py`

```python
@dataclass
class ScaffoldResult(_SerializableMixin):
    """Result of scaffold generation."""
    success: bool
    edits: list[PatchEdit] = field(default_factory=list)
    header_files: list[str] = field(default_factory=list)    # .h files created/modified
    source_files: list[str] = field(default_factory=list)     # .c files created/modified
    function_stubs: list[FunctionStub] = field(default_factory=list)
    error_info: str | None = None
    raw_response: str = ""
    compilation_output: str | None = None

@dataclass
class FunctionStub(_SerializableMixin):
    """A function stub from the scaffold, to be implemented independently."""
    name: str
    file_path: str              # Which .c file contains this stub
    signature: str              # Full C signature
    docstring: str              # Behavioral contract from scaffold
    start_line: int             # Line range in scaffolded file
    end_line: int
    dependencies: list[str]     # Other function names this may call
    header_file: str | None     # Which .h file declares this
```

### 2. New execute function — `execute_scaffold()` in `execute/scaffold.py` (new file)

```python
def execute_scaffold(
    context: ContextPackage,
    part_plan: PartPlan,
    llm: LoggedLLMClient,
    *,
    cumulative_diff: str | None = None,
) -> ScaffoldResult:
    """Generate compilable C scaffold from part plan."""
```

- Builds prompt from: ContextPackage + all steps in PartPlan (target files, descriptions)
- System prompt asks for `.h` files with guards, type definitions, function declarations with docstrings, and `.c` files with stubs
- Parses XML edit blocks (same format as implement)
- Returns ScaffoldResult with extracted FunctionStub list

### 3. Scaffold prompt — new entry in `SYSTEM_PROMPTS` dict

```python
SYSTEM_PROMPTS["scaffold"] = """You are Jane, a C code architect. Given a plan with implementation steps, generate a complete compilable scaffold.

Output format: XML edit blocks (same as implementation edits).

Requirements:
- .h files: #include guards, all struct/enum/typedef definitions, all function declarations with parameter names, docstring comments describing behavior/return values/error conditions/preconditions
- .c files: #include directives, function stubs with signature + docstring + minimal valid return (return 0; or return NULL;)
- The scaffold MUST compile with gcc -c -fsyntax-only (no linking)
- Every function that will be implemented must have a stub
- Docstrings are the implementation contract — be precise about behavior, edge cases, ownership semantics

Do not implement any function bodies. Stubs only."""
```

### 4. Compilation check — `validate_scaffold_compilation()` in `execute/scaffold.py`

```python
def validate_scaffold_compilation(
    files: list[str],
    repo_path: Path,
    *,
    compiler: str = "gcc",
    flags: str = "-c -fsyntax-only -Wall",
) -> tuple[bool, str]:
    """Run gcc on scaffolded files. Returns (success, output)."""
```

- Runs `gcc -c -fsyntax-only -Wall` on each `.c` file
- Returns (True, "") or (False, compiler_error_output)
- This is a binary check — the scaffold compiles or it doesn't

### 5. Per-function implementation — `execute_function_implement()` in `execute/scaffold.py`

```python
def execute_function_implement(
    stub: FunctionStub,
    scaffold_content: dict[str, str],   # file_path -> full scaffold source
    llm: LoggedLLMClient,
) -> StepResult:
    """Implement one function body from its scaffold stub."""
```

- Context per call: the `.h` file(s) + the `.c` file containing the stub + the stub's docstring
- Much smaller context than full ContextPackage (~3-5K tokens vs 20K+)
- Output: one search/replace edit replacing the stub body with the implementation
- The search string is the stub body; the replacement is the real implementation

### 6. Orchestrator integration — new scaffold pass in `runner.py`

Insert between part_plan and the code step loop:

```python
# --- SCAFFOLD PASS (C projects only) ---
if scaffold_enabled and is_c_project:
    scaffold_result = _run_scaffold_pass(ctx, part, part_plan, part_context, ...)
    if scaffold_result.success:
        # Apply scaffold edits
        scaffold_patch = apply_edits(scaffold_result.edits, repo_path)
        # Validate compilation
        compiled, output = validate_scaffold_compilation(...)
        if not compiled:
            # Scaffold failed — rollback and fall through to normal implement
            rollback_edits(scaffold_patch, repo_path)
            scaffold_result = None
        else:
            # Replace step loop with per-function implement loop
            for stub in scaffold_result.function_stubs:
                step_result = execute_function_implement(stub, scaffold_content, coding_llm)
                ...
```

The scaffold pass is **optional and fallback-safe**: if it fails (bad scaffold, compilation error), the orchestrator falls through to the existing step-by-step implement loop. This is not a "fallback" in the banned sense — it's a feature toggle with a degradation path that preserves the existing behavior when scaffold doesn't apply.

### 7. Config additions

```toml
[orchestrator]
scaffold_enabled = false        # Feature toggle — disabled by default
scaffold_compiler = "gcc"       # Compiler for scaffold validation
scaffold_compiler_flags = "-c -fsyntax-only -Wall"
```

Config classification:
- `scaffold_enabled`: **Optional** — default false. When true and target is C, scaffold pass runs.
- `scaffold_compiler`: **Optional** — default "gcc". Only read when scaffold_enabled=true. **Fail-fast**: if scaffold_enabled=true and compiler not found on PATH, raise hard error at config validation.
- `scaffold_compiler_flags`: **Optional** — default "-c -fsyntax-only -Wall".

### 8. C project detection

Per-part detection. If all target_files in the part_plan have `.c` or `.h` extensions, treat as C part. This allows mixed-language projects where some parts are C and others aren't.

### 9. Stub extraction via tree-sitter (decided)

After applying scaffold edits, parse the resulting `.c` files with **tree-sitter-c** to find all function definitions. Any function with a body matching the stub pattern (`{ return 0; }`, `{ return NULL; }`, `{ /* TODO */ return 0; }`) is a stub needing implementation. This is deterministic, uses existing AST infrastructure, and doesn't add LLM output format complexity. Requires adding tree-sitter-c grammar as a dependency.

### 10. Compilation is mandatory (decided)

If `scaffold_enabled = true` in config but `gcc` is not found on PATH, raise a hard error at config validation time. The scaffold is only useful when compilation-validated — an unvalidated scaffold propagates errors downstream. Aligns with the project's fail-fast principle.

## Files to Modify

| File | Change |
|------|--------|
| `src/clean_room_agent/execute/dataclasses.py` | Add `ScaffoldResult`, `FunctionStub` dataclasses |
| `src/clean_room_agent/execute/scaffold.py` | **NEW** — `execute_scaffold()`, `validate_scaffold_compilation()`, `execute_function_implement()` |
| `src/clean_room_agent/execute/prompts.py` | Add scaffold system prompt to `SYSTEM_PROMPTS`, add `build_scaffold_prompt()` |
| `src/clean_room_agent/orchestrator/runner.py` | Add scaffold pass between part_plan and step loop, per-function implement loop |
| `src/clean_room_agent/config.py` | Add scaffold config fields to default template |
| `planning/cli-and-config.md` | Document scaffold config fields and classification |

## Files NOT Modified

- `src/clean_room_agent/retrieval/` — no retrieval changes (scaffold uses same ContextPackage)
- `src/clean_room_agent/llm/` — no LLM changes (uses existing client)
- `src/clean_room_agent/execute/patch.py` — reuses existing edit/rollback
- `src/clean_room_agent/execute/parsers.py` — reuses existing XML edit parser

## Implementation Order

1. **Dataclasses** — `ScaffoldResult`, `FunctionStub` in dataclasses.py
2. **Scaffold prompt** — system prompt + `build_scaffold_prompt()` in prompts.py
3. **`execute_scaffold()`** — scaffold generation + parsing in new scaffold.py
4. **Compilation validation** — `validate_scaffold_compilation()` in scaffold.py
5. **Per-function implement** — `execute_function_implement()` in scaffold.py
6. **Orchestrator integration** — scaffold pass, per-function loop, fallback path
7. **Config** — scaffold config fields, default template update
8. **Tests** — unit tests for scaffold generation, compilation check, per-function implement, orchestrator integration

## Verification

- Run existing test suite: `pytest tests/` (must pass — scaffold disabled by default)
- Unit tests for `execute_scaffold()` with mock LLM producing valid C scaffold
- Unit tests for `validate_scaffold_compilation()` with real gcc (requires gcc on PATH)
- Unit tests for `execute_function_implement()` with mock scaffold content
- Integration test: scaffold → compile → per-function implement → compile again → tests pass
- Manual: enable scaffold in config, run `cra solve` on a C task, verify scaffold compiles and functions get implemented independently

## Relationship to Binary Decomposition Plan

Scaffold benefits from model tiers (scaffold → 4B, per-function → 1.7B), but works independently with the existing 2-role system (scaffold → reasoning, per-function → coding). **Implement binary decomposition first** (smaller scope, retrieval-only), then scaffold (larger scope, orchestrator changes).
