# Scaffold-Then-Implement: Exploiting C's Dependency Graph

Date: 2026-02-27

## Decision

Insert a scaffold step between planning and code generation. The scaffold produces the file structure — header declarations, function signatures, struct definitions, docstrings describing behavior and return values, stub implementations. Each function is then implemented by an independent LLM call against the scaffold. Implementation calls are parallel, independent, and go to the smallest capable model.

## Pipeline

```
Plan (4B) → Scaffold (4B) → Implement[0..N] (1.7B, parallel) → Compile → Test
```

### Plan

Decomposes the task into parts and steps. Existing meta-plan / part-plan architecture. Output: what to build, which files, which functions.

### Scaffold

Produces compilable code with empty function bodies. Specifically:

- `.h` files: `#include` guards, struct/enum/typedef definitions, function declarations with parameter names, docstring comments describing behavior, return values, error conditions, and preconditions.
- `.c` files: `#include` directives, function stubs (signature + docstring + minimal valid return like `return 0;` or `return NULL;`).

The scaffold is a compilable artifact. It passes `gcc -fsyntax-only`. It defines the complete interface contract before any implementation exists.

This is the reasoning-heavy step. Choosing the right function boundaries, the right struct layouts, the right behavioral contracts in docstrings — this is where the 4B model's capacity matters. A bad scaffold produces bad implementations regardless of the implementation model's quality.

### Implement (per-function, parallel)

Each function implementation is a separate LLM call. The call receives:

- The scaffold (all `.h` files + the `.c` file containing the target function's stub)
- The target function's docstring (what to implement)
- Referenced type definitions (structs, enums used in the signature)
- Any helper function signatures the implementation may call

The call produces: one function body replacing the stub.

These calls are:
- **Independent.** Implementing `parse_token()` doesn't need the implementation of `scan_next()` — only its signature and docstring.
- **Parallel.** N functions → N simultaneous calls. No sequential dependency between function implementations within the same file.
- **Minimal context.** The scaffold is compact (signatures + docstrings, no implementation bodies). A 1.7B model with 32K context holds the entire scaffold plus the implementation target easily.
- **Sent to the 1.7B.** This is the high-volume step. N calls per task, each producing one function body. The smaller model handles the concrete code generation work.

### Compile

`gcc -Wall -Werror` on the assembled files. Binary outcome: compiles or doesn't. Compilation errors point to specific functions — the error attribution is clean.

### Test

Function-level tests generated from the scaffold's docstrings (a separate step, also parallelizable). Each test exercises one function's contract. Binary outcome per function.

## Why This Works for C

### C's dependency graph is explicit and simple

Headers declare, source files implement. The scaffold step maps directly onto C's compilation model — writing the `.h` file first IS how experienced C programmers work. The header is the contract. There is no hidden dependency resolution, no import magic, no runtime dispatch ambiguity. If the header compiles, the interface contract is well-defined.

### The scaffold IS the test specification

The docstrings in the scaffold describe: what the function does, what it returns, under what error conditions. This is sufficient to generate tests before any implementation exists. Test generation can work from the scaffold alone, and the tests become the validation oracle for the implementation calls.

### Minimal context per implementation call

A C header file with 20 function signatures and docstrings might be 200 lines / ~2K tokens. The implementation target (one function stub + its docstring) adds maybe 20 lines. Referenced structs add maybe 50 lines. Total context per implementation call: ~3-5K tokens. This is 10-15% of the 1.7B's 32K window. The context is nearly 100% signal.

Compare: the current implement step sends the full ContextPackage (potentially 20K+ tokens of multi-file source) and asks the model to produce search/replace edits across multiple files. The signal-to-noise ratio is much lower.

### Error isolation

A bad implementation of function A doesn't contaminate function B. If `parse_token()` has a bug, `scan_next()` still works (assuming `parse_token()`'s signature and documented contract are correct). Bugs are isolated to individual functions, which means:

- Debugging points to one function, one LLM call, one context window.
- Re-generation retries one call, not the entire implementation.
- The audit trail in the raw DB traces each function's implementation to a specific LLM call with specific input.

## Connection to Binary Decomposition

This is the binary decomposition principle applied to code generation:

- **Does function X compile?** Binary.
- **Does function X pass its tests?** Binary.
- **Is function X's implementation correct given its docstring contract?** Binary (and auditable by reading the docstring + implementation side by side).

Each function implementation is an independent binary-outcome decision. The aggregate (does the program work?) composes from the individual function outcomes. Errors decompose cleanly — a failing test points to a specific function, which points to a specific LLM call.

## Connection to Model Tiers

| Step | Model | Why |
|---|---|---|
| Plan | 4B | Task decomposition requires broad reasoning |
| Scaffold | 4B | Function boundary design and behavioral contracts are the hard reasoning problem |
| Implement[0..N] | 1.7B | Each call is narrow: one function body from one docstring. High volume, parallelizable. |
| Test generation | 1.7B | Each call generates tests for one function from its docstring. Same pattern. |

The 4B does the thinking (2 calls: plan + scaffold). The 1.7B does the volume work (N calls: implementations + tests). Total 4B usage is constant per task. Total 1.7B usage scales with task size but each call is cheap.

## Risks

1. **Scaffold quality is the single point of failure.** A bad function decomposition, wrong struct layout, or vague docstring propagates to all downstream implementation calls. The scaffold step must be the most carefully prompted and validated step. Mitigation: scaffold compilation check (syntactic validity) + human review protocol for scaffold quality.
2. **Cross-function invariants.** Some implementations need to maintain invariants across functions (e.g., a linked list's `insert` and `delete` must agree on node layout). The scaffold's struct definitions and docstrings must make these invariants explicit. If they're implicit, implementations will diverge.
3. **Implementation calls may need sibling context.** In practice, implementing `foo()` sometimes requires seeing how `bar()` is implemented (not just its signature) — especially when they share internal state or have coupled error handling. The scaffold must surface these couplings as explicit documentation, or the implementation step needs optional sibling function bodies as additional context.
4. **Doesn't generalize to Python.** Python's dependency graph is implicit (imports resolve at runtime, duck typing, no header files). The scaffold pattern relies on C's explicit interface declarations. For Python code generation, a different intermediate representation would be needed. This is fine — the project is targeting C.
