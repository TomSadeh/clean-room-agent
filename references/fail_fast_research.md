# Fail-fast repositories for LoRA training on commit histories

**The best training candidates are not web frameworks or HTTP libraries — they are parsers, type checkers, validation libraries, and projects by a small number of opinionated developers.** After examining 60+ repositories across Python and TypeScript, the strongest fail-fast signal comes from a predictable cluster: Hynek Schlawack's projects (structlog, attrs, svcs, stamina), parser/compiler tools (Black, LibCST, parso), and validation libraries that practice what they preach internally (cattrs, strictyaml, typeguard). These repos share a common DNA: custom exception hierarchies, rich contextual error messages with interpolated variables, narrow `except` clauses, and a refusal to silently swallow failures. For a 20–30 repo training corpus, the recommended mix spans 6+ domains to prevent the model from conflating domain idioms with the target error-handling style.

---

## 1. Which domains actually produce fail-fast code

The initial hypotheses about fail-fast domains are **mostly correct, with important exceptions**. Compilers, parsers, type checkers, and data validation libraries are the strongest natural producers. CLI frameworks and configuration libraries are more nuanced than expected — their boundary layers are inherently defensive even when internal code is strict.

**Strongly fail-fast domains (confirmed):**
Parsers and compilers must be correct or loudly fail — there is no meaningful "graceful degradation" for a malformed AST. Type systems and static analysis tools share this property. Data validation libraries (when they practice strict patterns internally, not just in their API) are excellent because their entire purpose is rejecting bad input. Runtime type checkers like typeguard and beartype are almost pure fail-fast by definition. Serialization libraries that default to strict mode (msgspec, orjson) also fit well.

**Domains that seem fail-fast but are actually defensive:**
Web frameworks (Django, Flask, FastAPI) are the biggest trap. They have clean architecture, good separation of concerns, and sophisticated exception hierarchies — but every one of them has a top-level catch-all handler that prevents server crashes. **HTTP client libraries** (requests, httpx) are similarly deceptive: they have well-designed exception classes, but default behavior returns HTTP errors as normal response objects rather than raising. Message queues and distributed systems (Celery) are heavily defensive by necessity. Configuration libraries are mixed — they often use `.get()` with defaults everywhere because config values are inherently optional.

**Missing from the original hypotheses:**
- **Runtime type checking tools** (typeguard, beartype) — perhaps the purest fail-fast category
- **Cryptography/security libraries** — must fail loudly on any error
- **Structured logging libraries** — structlog is both a fail-fast practitioner and a tool for making failures visible
- **Monadic/Result-type libraries** (dry-python/returns) — encode fail-fast into the type system itself

**CLI frameworks deserve a split verdict.** Click has an exemplary exception hierarchy (`UsageError`, `BadParameter`, `FileError`) with rich context and `raise ... from e` chaining. But its `CliRunner.invoke` catches all exceptions broadly, and `standalone_mode` converts exceptions to formatted error messages. The internal library code is fail-fast; the framework boundary is defensive. This makes CLI frameworks good for learning exception *design* but poor for learning error *propagation*.

---

## 2. Ranked repository recommendations

### Tier 1: Excellent training data (strong fail-fast + good commits + right size)

**1. strictyaml** — `crdoconnor/strictyaml` — Python — MIT
The entire library is a fail-fast manifesto. It rejects ambiguous YAML features (flow mappings, tag tokens, anchors) with **18+ specific `*Disallowed` exception types**, each carrying line numbers and code snippets. The `expecting_but_found()` pattern produces errors like `"when expecting a float / found arbitrary text / in '<unicode string>', line 5, column 1"`. Error messages include the offending source with a caret pointing to the exact position. The library's README explicitly documents its philosophy of rejecting ambiguity. **~15 source files**, comprehensive story-based test suite, good commit history. The strongest philosophical alignment of any repo examined.

**2. attrs** — `python-attrs/attrs` — Python — MIT
Hynek Schlawack's flagship project has **8+ specific exception types** including `FrozenInstanceError`, `AttrsAttributeNotFoundError`, `NotAnAttrsClassError`, and `UnannotatedAttributeError`. Error messages are rich f-strings: `"No mandatory attributes allowed after an attribute with a default value or factory. Attribute in question: {a!r}"`. Both the external API and internal `_make.py` code follow fail-fast patterns. Functions document their `Raises:` in docstrings. **~15 source files**, excellent commit hygiene, extensive test suite, **5.7k stars**.

**3. cattrs** — `python-attrs/cattrs` — Python — MIT
Uses **Python 3.11 ExceptionGroups** for collecting multiple validation errors: `"While structuring Foo (2 sub-exceptions)"`. Path-annotated errors via `__notes__` with `AttributeValidationNote("Structuring class Foo @ attribute foo", "foo", __c_type_foo)`. Generated structuring code uses `o['field']` (direct dict access, raises `KeyError` on missing required fields). The `transform_error()` utility converts exception trees to `"{description} @ {path}"` format. Historical commits show progression from generic `raise Exception(...)` to custom `ForbiddenExtraKeyError` — ideal for training on error-handling improvements. **~20 source files**, MIT license, active maintenance.

**4. LibCST** — `Instagram/LibCST` — Python — MIT
The most deliberate exception design of any parser examined. `ParserSyntaxError` is marked `@final` and explicitly does **not** inherit from Python's `SyntaxError` (documented reason: "Python's may raise a SyntaxError for any number of reasons, potentially leading to unintended behavior"). Carries `message`, `raw_line`, `raw_column`, `editor_line`, `editor_column`. Internal `_InternalSyntaxError` sentinel is documented as "should never be visible to the end-user." **1,788+ tests** including Hypothesis fuzz testing and Pyre type-checking. **~150 source files**, excellent descriptive commits.

**5. structlog** — `hynek/structlog` — Python — Apache-2.0/MIT dual
Hynek Schlawack's structured logging library embodies the philosophy that errors are information. The `ExceptionRenderer` class replaces `exc_info` with rendered `exception` fields. Immutable logger pattern (`.bind()` returns new instances). Type-safe async detection — calling `Container.get()` on an async factory raises `TypeError` rather than silently failing. **100% test coverage requirement**, signed releases with attestations, **~50 source files**. The sweet spot of size, quality, and opinionated error handling.

**6. Black** — `psf/black` — Python — MIT
Custom exception hierarchy (`InvalidInput`, `NothingChanged`, `CannotTransform`) plus **Rust-style `Ok`/`Err` Result types** in `black/rusty.py`. AST safety verification: reformatted code is checked for AST equivalence, raising `INTERNAL ERROR` with detailed context on mismatch. Narrow except clauses even in the blackd server — the only broad `except Exception` is at the outermost HTTP handler with `logging.exception()`. **~50+ source files**, **41.4k stars**, very active maintenance, comprehensive test suite with fuzz testing.

**7. Hypothesis** — `HypothesisWorks/hypothesis` — Python — MPL-2.0
Rich exception hierarchy with 10+ types including `Frozen`, `InvalidArgument`, `FlakyFailure` (inherits `ExceptionGroup`), and `UnsatisfiedAssumption` ("If you're seeing this error something has gone wrong"). Contextual error messages with f-strings, consistent `from err` exception chaining, active traceback trimming (`err.with_traceback(get_trimmed_traceback())`). **Caveat: MPL-2.0 with Exhibit B** — weak copyleft that triggers on distribution of modified source files. Using as training data is a legally grey area; consult counsel. The Python testing portion is substantial and high-quality.

**8. typeguard** — `agronholm/typeguard` — Python — MIT
Raises `TypeCheckError` immediately on type violations. Three instrumentation modes (decorator, import hook, explicit check). Checks function arguments, return values, and annotated local variables. Test configuration uses `xfail_strict = true` and `filterwarnings = ["error"]` — fail-fast even in testing. **~30–50 files**, **1,700 stars**, very active (v4.4.4 released June 2025).

**9. svcs** — `hynek/svcs` — Python — MIT
Modern service locator with fail-fast on misuse: `Container.get()` on an async factory raises `TypeError`. `ResourceWarning` when containers are garbage-collected with pending cleanups — no silent resource leaks. Cleanup methods catch and log exceptions (the one place graceful degradation is appropriate). **100% coverage requirement**, small focused codebase, same high standards as structlog.

**10. parso** — `davidhalter/parso` — Python — MIT
Architecturally sophisticated dual-mode parser: `error_recovery=False` raises `ParserSyntaxError` (fail-fast), `error_recovery=True` enables IDE completion mode. Tests validate error cases with `pytest.raises(ParserSyntaxError)`. This dual-mode design is pedagogically valuable — it demonstrates how to build fail-fast-by-default with optional recovery. **~30–40 source files**, good commit quality.

### Tier 1.5: Very good candidates with minor caveats

**11. mypy** — `python/mypy` — Python — MIT
`CompileError` carries structured data: `messages: list[str]`, `module_with_blocker: str | None`, `use_stdout: bool`. Textbook fail-fast pattern. **Caveat: very large** (~100+ source files, possibly exceeding the ideal range). Core error handling is excellent but the codebase may be too large to use in full.

**12. Werkzeug** — `pallets/werkzeug` — Python — BSD-3-Clause
Gold standard for exception hierarchies with contextual data. `MethodNotAllowed` carries `valid_methods`, `ArgumentValidationError` carries `missing`, `extra`, and `extra_positional` sets, `ImportStringError` carries `import_name` and wrapped `exception`. Mixed Python exceptions + HTTP exceptions ("KeyError that is also a BadRequest"). **Caveat**: as a web toolkit, some boundary-layer patterns are defensive.

**13. Click** — `pallets/click` — Python — BSD-3-Clause
`ClickException` → `UsageError` → `BadParameter`, with contextual `ctx` parameter, `show()` methods, and `raise FileError(self.name, hint=e.strerror) from e` pattern. Excellent exception hierarchy design. **Caveat**: CLI runner catches all exceptions broadly; `standalone_mode` prevents propagation.

**14. glom** — `mahmoud/glom` — Python — BSD-3-Clause
Raises `PathAccessError` with detailed messages telling you exactly which step in a nested path failed. `GlomError` base class for all errors. **~30–50 files**, **2.1k stars**, tested on Python 3.7–3.14 + PyPy3. Optional `default` parameter allows fallback (opt-in graceful degradation, fail-fast by default).

**15. stamina** — `hynek/stamina` — Python — MIT
Retry library that embodies narrow exception catching as API design: `@stamina.retry(on=httpx.HTTPError, attempts=3)`. Callable exception filtering for granular control. Small, focused, opinionated. Ideal as supplementary training data for error handling in retry/resilience contexts.

### Tier 1 TypeScript/JavaScript

**16. Zod** — `colinhacks/zod` — TypeScript — MIT
`.parse()` throws `ZodError` immediately (fail-fast default). `.safeParse()` returns discriminated union `{ success: true, data: T } | { success: false, error: ZodError }` (Result-type pattern). `ZodIssueCode` enum includes `invalid_type`, `invalid_literal`, `unrecognized_keys`, etc. Internal validation throws on unexpected states: `"Validation failed but no issues detected"`. Requires `strict: true` in tsconfig. **41.9k stars**, excellent test suite, moderate codebase size.

**17. arktype** — `arktypeio/arktype` — TypeScript — MIT
Extreme TypeScript strictness: tests use `strictNullChecks: true` and `exactOptionalPropertyTypes: true`. `toJsonSchema()` throws on incompatible features by default, with opt-in fallback. Type-level benchmarking via `attest` tool — `ATTEST_benchErrorOnThresholdExceeded` fails CI on performance regression. Active single maintainer with very high commit quality.

**18. io-ts** — `gcanti/io-ts` — TypeScript — MIT
`decode()` returns `Either<Errors, A>` — never throws. Built on fp-ts `Either` discriminated unions. `PathReporter` formats error paths. Small, focused codebase. Custom error messages via `t.failure(input, context, 'custom message')`. Clean Result-type patterns throughout.

**19. Chevrotain** — `Chevrotain/chevrotain` — TypeScript — Apache-2.0
**18+ specific `LexerDefinitionErrorType` values** (MISSING_PATTERN, INVALID_PATTERN, SOI_ANCHOR_FOUND, etc.). Structured error objects with `message`, `type`, `tokenTypes` fields. Error messages include links to resolution docs. Separate `IParserDefinitionError` types. **Caveat**: does not use `strict: true` in tsconfig. **~9,000 LOC**, good monorepo structure.

### Tier 2: Good supplementary candidates

| Repo | Language | License | Notes |
|------|----------|---------|-------|
| **bandit** (`PyCQA/bandit`) | Python | Apache-2.0 | Narrow `except OSError:` → `ConfigError`, documented `:raises:` |
| **coverage.py** (`nedbat/coveragepy`) | Python | Apache-2.0 | Specialized exceptions (v6.2+), careful edge case handling |
| **pytest** (`pytest-dev/pytest`) | Python | MIT | Structured exit codes, `ExceptionInfo` wrapping — but very large |
| **SQLAlchemy** (`sqlalchemy/sqlalchemy`) | Python | MIT | Gold-standard exception hierarchy — but enormous codebase |
| **Jinja2** (`pallets/jinja`) | Python | BSD-3-Clause | `TemplateSyntaxError` with line numbers — large |
| **beartype** (`beartype/beartype`) | Python | MIT | Rich error messages — possibly too large |
| **returns** (`dry-python/returns`) | Python | BSD-2-Clause | Monadic Result types, `@safe` decorator — medium-large |
| **environ-config** (`hynek/environ-config`) | Python | Apache-2.0 | `MissingEnvValueError`, aggregates all missing secrets |
| **Effect** (modules) (`Effect-TS/effect`) | TypeScript | MIT | `Cause<E>` "lossless error model" — extract individual modules |
| **tRPC** (packages) (`trpc/trpc`) | TypeScript | MIT | `TRPCError` with typed codes — extract `@trpc/server` |
| **Curio** (`dabeaz/curio`) | Python | BSD | `UncaughtTimeoutError` teaches good patterns — **abandoned** |
| **rope** (`python-rope/rope`) | Python | LGPL-3.0 | Good hierarchy but mixed internal patterns; **LGPL** |

---

## 3. Identifying fail-fast repos programmatically

### AST-based heuristic scorer

The most predictive automated approach combines **AST analysis of exception handling patterns** with **linter configuration inspection**. A scoring system built on these metrics can rank repositories before manual review:

| Metric | Weight | Measurement | Fail-fast signal |
|--------|--------|-------------|-----------------|
| Bare except ratio | 20% | `except:` / total handlers | Low = fail-fast |
| Blind except ratio | 15% | `except Exception:` / total handlers | Low = fail-fast |
| Try-except-pass count | 15% | AST count of try/except/pass | Low = fail-fast |
| Specific exception ratio | 15% | Specific handlers / total handlers | High = fail-fast |
| dict["key"] vs .get() ratio | 10% | Direct / (direct + .get()) | High = fail-fast |
| Assert density | 10% | assert statements / KLOC | High = fail-fast |
| Raise density | 10% | raise statements / KLOC | High = fail-fast |
| Exception chaining (`from`) | 5% | `raise X from Y` / total raises | High = fail-fast |

A repo scoring **75–100** on this scale is a strong candidate. Scores of **50–74** indicate mixed patterns worth manual review. Below 50 is likely defensive.

### Ruff rules that correlate with fail-fast style

Projects that enable these rules in their `ruff.toml` or `pyproject.toml` are self-selecting for fail-fast discipline: **E722** (bare-except), **BLE001** (blind-except), **S110** (try-except-pass), **S112** (try-except-continue), **TRY203** (useless-try-except), and the full **TRY** category from tryceratops. Searching GitHub for `filename:ruff.toml "BLE001"` or `filename:pyproject.toml "[tool.mypy]" "strict"` efficiently surfaces quality-conscious repos.

### Practical approach for scale

**Semgrep** can scan repos with custom YAML rules matching the exact code patterns you're seeking — more flexible than AST parsing and supports 17+ languages. For bulk screening, combine GitHub Code Search queries (e.g., `language:python "except:" NOT "except BaseException"` to find bare-except offenders, or `filename:mypy.ini "strict = true"` to find strict-mode projects) with automated Ruff analysis on cloned repos. **Radon** measures cyclomatic complexity (high complexity often correlates with deep defensive nesting) and **Wily** tracks complexity over git history — useful for verifying that a repo's style is consistent across its lifetime rather than a recent cleanup.

The academic **Style2Code** framework (2025) demonstrates that code style can be encoded as a learnable representation using contrastive learning with InfoNCE loss. This suggests a bootstrapping approach: manually label 50–100 functions as fail-fast or defensive, train a lightweight classifier, then use it to screen thousands of repos automatically.

---

## 4. Anti-patterns: repos that look good but train the wrong style

**Django** is the most dangerous false positive. It has clean architecture, great documentation, and a sophisticated exception system — but its template engine **intentionally silences errors** to prevent user-facing crashes, its middleware uses broad `except Exception` blocks, and its ORM has numerous silent fallback behaviors. Training on Django commits would embed deeply defensive patterns.

**Flask** follows the same trap. Miguel Grinberg (Flask contributor) states plainly: "Flask catches all errors, so your application will never crash due to missing to catch an error." Flask-SQLAlchemy auto-rolls back sessions on any database error. The `full_dispatch_request()` implements catch-all exception handling.

**Requests** by Kenneth Reitz has a well-designed exception hierarchy (`ConnectionError`, `Timeout`, `HTTPError`) — but **HTTP errors (4xx, 5xx) are returned as normal response objects** without raising. `raise_for_status()` is opt-in. The default behavior is fundamentally defensive: errors pass silently unless explicitly requested.

**Invoke** has acknowledged broken error handling — issue #269 states "Error handling is not implemented in Invoke yet." Missing config files cause silent skips. **python-dotenv** returns `True` even when the specified `.env` file doesn't exist. Both are anti-matches despite being popular.

**Celery** is heavily defensive by necessity — task queues must survive worker crashes, network failures, and arbitrary exceptions. Its retry mechanisms, error callbacks, and broad exception suppression patterns are correct for its domain but toxic for fail-fast training.

**The general rule**: any project whose primary job is keeping a long-running process alive (web servers, task queues, message brokers, UI frameworks) will be defensive at its boundaries. This is correct engineering — but wrong training data.

---

## 5. Curating a training corpus of 20–30 repos

### Recommended distribution

The corpus should span **6+ distinct domains** with no more than 20% from any single domain. Based on the research, here is an optimal 25-repo corpus:

**Parsers & compilers (5 repos):** strictyaml, Black, LibCST, parso, Chevrotain (TS)

**Validation & type checking (5 repos):** attrs, cattrs, typeguard, Zod (TS), arktype (TS)

**Structured data & serialization (3 repos):** glom, msgspec, io-ts (TS)

**Developer tooling (4 repos):** structlog, Hypothesis, Click, bandit

**Small focused libraries (4 repos):** svcs, stamina, environ-config, MarkupSafe

**Frameworks with good error design (4 repos):** Werkzeug, coverage.py, Jinja2, returns

### Commit filtering strategy

Research from **CommitPackFT** (NeurIPS 2023 Workshop) shows that filtering for single-file commits with imperative-mood messages starting with verbs like "Fix," "Add," or "Verify" produces the highest-quality instruction-completion pairs. Only **5,000–10,000 examples** from 20–30 diverse repos are likely sufficient for a style-focused LoRA. Filter out merge commits, version bumps, dependabot updates, and documentation-only changes. Filter *in* commits that add/improve error handling, replace broad exceptions with specific ones, add custom exception classes, or add input validation — search commit messages for "fix error handling," "add validation," "raise instead of return," "strict," "assert."

### Should you include negative examples?

**Primary approach: positive examples only.** For LoRA fine-tuning, Sebastian Raschka's research shows data quality matters far more than quantity, and defensive code that is "correct but wrong style" is subtle enough to confuse the model. If using DPO-style preference training, the **CodeFavor** approach — using pre-commit code as rejected and post-commit code as accepted — is directly applicable. The cattrs repo has commits showing progression from `raise Exception(f"Literal {type} not equal to {val}")` to custom `ForbiddenExtraKeyError`, which makes ideal before/after training pairs.

### Preventing domain overfitting

The biggest risk is the model learning parser idioms (visitor pattern, AST manipulation) or attrs-ecosystem conventions as part of "fail-fast style." Mitigation: include repos from authors across at least 5 different organizations, vary codebase sizes from 1K to 50K LOC, mix library code with tool code, and validate on held-out repos from domains not in the training set. The proposed heuristic scorer should be run on all candidates to verify they score consistently high on fail-fast metrics.

---

## Gaps and limitations

**No strong fail-fast repos were found in infrastructure-as-code** (Terraform CDK, Pulumi Python SDK). These tend toward defensive patterns because infrastructure operations are inherently fallible. **Scientific computing** repos (NumPy, SciPy) are also defensive internally despite being correctness-critical — they prioritize numerical stability over loud failure. **No Python repos were found that explicitly document fail-fast philosophy in CONTRIBUTING.md** — the principle manifests as architectural decisions rather than written policy. The TS/JS ecosystem has fewer pure-Python-style fail-fast candidates because many of the best tools (Biome, oxc, SWC, esbuild) are written in Rust or Go with thin JS bindings, leaving insufficient TypeScript source for training. **Zod, arktype, and io-ts** are the strongest TS candidates with adequate source code volume.

## Conclusion

The fail-fast style concentrates in a surprisingly small number of developers and domains. **Hynek Schlawack's projects** (attrs, cattrs, structlog, svcs, stamina, environ-config) represent the single most valuable cluster — consistently opinionated, well-tested, MIT-licensed, and in the ideal size range. **Parser and compiler tools** (strictyaml, Black, LibCST, parso) are the most reliable domain signal. **Runtime type checkers** (typeguard, beartype) are an underappreciated category — nearly pure fail-fast by definition. The key insight for corpus curation is that fail-fast style is fundamentally about a *relationship to errors* — treating them as information rather than threats — and this philosophy crosses domain boundaries when practiced by the right developers. A 25-repo corpus spanning these developers and domains, filtered to 5,000–10,000 high-quality commit pairs, should produce a LoRA adapter that generates code crashing informatively rather than degrading silently.