# Test Suite Audit: `.get()` to `dict[key]` Conversion

**Date**: 2026-02-27
**Scope**: All test files under `tests/` audited against production code after ~70 `.get(key, default)` calls were converted to direct `dict[key]` access for fail-fast behavior.
**Baseline**: 1289/1289 tests pass.

---

## Summary

| Classification | Count | Fixed | Description |
|---|---|---|---|
| **BROKEN** | 4 | 4 | Will crash in production or silently produces wrong results |
| **STALE** | 14 | 4 | Tests exercise dead behavior or have misleading names |
| **FRAGILE** | 17 | 2 | Pass now but break on code reorder, incomplete LLM response, or mock strategy change |

---

## BROKEN Findings

### B1. `trace.py:46` — `call["thinking"]` KeyError on non-thinking models — FIXED

- **Test**: `tests/test_trace.py:10-31` (`_make_call` helper always provides `thinking=""`)
- **Production**: `src/clean_room_agent/trace.py:46` — `call["thinking"]`
- **Upstream**: `src/clean_room_agent/llm/client.py:214-215` — `"thinking"` only added when `response.thinking is not None`
- **Issue**: `LoggedLLMClient.flush()` returns call dicts WITHOUT a `"thinking"` key for non-thinking models (the common case). The success-path record (client.py:206-213) omits it entirely. `trace.py:46` does `call["thinking"]` which will `KeyError` on the first LLM call with a non-thinking model.
- **Why tests pass**: `_make_call()` always provides `thinking=""`, masking the mismatch.
- **Fix**: Either restore `.get("thinking", "")` in trace.py (thinking is structurally optional — depends on model capability), or fix `LoggedLLMClient` to always emit `"thinking": ""` in the success record.

### B2. `trace.py:50` — `call["error"]` KeyError on successful LLM calls — FIXED

- **Test**: `tests/test_trace.py:10-31` (`_make_call` helper always provides `error=""`)
- **Production**: `src/clean_room_agent/trace.py:50` — `call["error"]`
- **Upstream**: `src/clean_room_agent/llm/client.py:206-213` — success record has no `"error"` key; only error path (line 202) includes it
- **Issue**: Every successful LLM call produces a record without `"error"`. `trace.py:50` will `KeyError` on every one.
- **Why tests pass**: `_make_call()` always provides `error=""`.
- **Fix**: Same as B1 — either restore `.get("error", "")` in trace.py (error is structurally absent on success), or fix `LoggedLLMClient` to always emit `"error": ""`.

### B3. `runner.py:942-950` — `apply_edits` `success=False` not checked in code step loop — FIXED

- **Test**: All `TestRunOrchestrator` tests mock `apply_edits` returning `PatchResult(success=True)`
- **Production**: `src/clean_room_agent/orchestrator/runner.py:942-950`
- **Production**: `src/clean_room_agent/execute/patch.py:100-101` — returns `PatchResult(success=False)` on validation errors
- **Issue**: Comment at runner.py:942 says "raises on failure — no success flag check needed." But `apply_edits` returns `PatchResult(success=False, error_info=...)` on validation errors (patch.py:101) WITHOUT raising. When this happens, runner.py:946 calls `update_run_attempt_patch(raw_conn, attempt_id, True)` — marking the patch as applied when it was not — then sets `step_success = True` at line 949. This is a production logic bug.
- **Why tests pass**: All mocks return `success=True`.
- **Fix**: Add `if not patch_result.success: raise RuntimeError(patch_result.error_info)` after line 943, or check the flag and handle the failure path. Add a test with `apply_edits` returning `success=False`.

### B4. `runner.py:1241-1247` — Same `apply_edits` gap in test step loop — FIXED

- **Production**: `src/clean_room_agent/orchestrator/runner.py:1241-1247`
- **Issue**: Identical to B3 but in the test step loop. Comment at line 1240 says "Both raise on failure — no success flag checks." Same incorrect assumption.
- **Fix**: Same as B3.

---

## STALE Findings

### S1. `test_audit_a1_a12.py:198` — Test name "empty returns default" but provides full config

- **Test**: `tests/test_audit_a1_a12.py:198-201` — `test_require_environment_config_empty_returns_default`
- **Production**: `src/clean_room_agent/config.py:151` — `config["environment"]`
- **Issue**: Was changed from `require_environment_config({})` to `require_environment_config({"environment": {"coding_style": _DEFAULT_CODING_STYLE}})`. Now tests a tautology (provides default, asserts default). The original missing-section behavior is now a `KeyError` crash, and the docstring (config.py:142-146) still says "Returns defaults if absent."
- **Fix**: Either (a) rename test to `test_require_environment_config_with_explicit_defaults` and add a separate `test_missing_environment_section_raises_keyerror`, or (b) restore `.get()` and update docstring if the section should genuinely be optional.

### S2. `test_environment.py:125` — Test name "no environment section" but provides one

- **Test**: `tests/test_environment.py:125-129` — `test_no_environment_section_defaults`
- **Production**: `src/clean_room_agent/environment.py:88` — `config["environment"]`
- **Issue**: Config includes `"environment": {"coding_style": "development"}`. Name implies missing-section testing but actually tests the happy path (already covered by `test_basic_build`).
- **Fix**: Delete (duplicate) or rename and rewrite to test `KeyError` on missing section.

### S3. `test_environment.py:131` — Test name "no testing section" but provides one

- **Test**: `tests/test_environment.py:131-135` — `test_no_testing_section_empty_framework`
- **Production**: `src/clean_room_agent/environment.py:83` — `config["testing"]`
- **Issue**: Config includes `"testing": {"test_command": ""}`. Same pattern as S2.
- **Fix**: Same as S2.

### S4. `test_router.py:119` — Test name "default temperature is zero" but temperature is required

- **Test**: `tests/llm/test_router.py:119-122` — `test_default_temperature_is_zero`
- **Production**: `src/clean_room_agent/llm/router.py:65` — `models_config["temperature"]`
- **Issue**: Temperature is now a required key (direct `[]` access). There is no default. The test asserts `0.0` from the explicit config, not from a default.
- **Fix**: Rename to `test_temperature_zero_passthrough`.

### S5. `test_enrichment.py` — No happy-path test; `public_api_surface` gap — FIXED (validation)

- **Test**: `tests/llm/test_enrichment.py` (entire file)
- **Production**: `src/clean_room_agent/llm/enrichment.py:156,179` — `parsed["public_api_surface"]`
- **Production**: `src/clean_room_agent/llm/enrichment.py:135` — `_REQUIRED_ENRICHMENT_FIELDS` does NOT include `public_api_surface`
- **Issue**: No test exercises the happy path (LLM returns valid JSON, result stored). `public_api_surface` is accessed via `dict[key]` but not validated in `_REQUIRED_ENRICHMENT_FIELDS`. An LLM response missing this field passes validation but crashes at line 156 with a raw `KeyError` instead of a structured `ValueError`.
- **Fix**: Add `"public_api_surface"` to `_REQUIRED_ENRICHMENT_FIELDS`. Add a happy-path test.

### S6. `test_pipeline.py:262` — `test_missing_config_raises` passes for wrong reason

- **Test**: `tests/retrieval/test_pipeline.py:262` — `test_missing_config_raises`
- **Production**: `src/clean_room_agent/retrieval/preflight.py:25` (now raises before pipeline runs)
- **Issue**: Passes `config=None` and matches `"No config file found"`. The error now comes from `require_models_config` in the preflight check, not from pipeline logic. The test accidentally still matches the same error message.
- **Fix**: Rename to `test_missing_config_caught_by_preflight` or similar.

### S7. `test_context_assembly.py:149` — `test_missing_file_skipped` tests R2 exclusion, not missing file

- **Test**: `tests/retrieval/test_context_assembly.py:149-159` — `test_missing_file_skipped`
- **Production**: `src/clean_room_agent/retrieval/context_assembly.py:129-175`
- **Issue**: Creates `ScopedFile` with `file_id=99` but provides NO `classified_symbols` for it. Passes because R2 default-deny excludes the file (no classified symbols), NOT because the file is missing from disk. Never reaches disk read. If it did, it would now raise `RuntimeError`.
- **Fix**: Rename to `test_file_without_classified_symbols_excluded` (note: there is already a test at line 161 that tests this explicitly — this test may be a duplicate).

### S8. `test_context_assembly.py` — No test for `_require_logged_client` negative path

- **Test**: `tests/retrieval/test_context_assembly.py` (entire file)
- **Production**: `src/clean_room_agent/retrieval/context_assembly.py:344-350`
- **Issue**: `_require_logged_client` is new code. It raises `TypeError` if LLM client lacks `flush`. No test covers the negative path.
- **Fix**: Add test passing `Mock(spec=LLMClient)` and asserting `TypeError`.

### S9. `test_runner.py:531` — `test_single_pass_failure` mocks impossible `success=False` — FIXED

- **Test**: `tests/orchestrator/test_runner.py:531` — `test_single_pass_failure`
- **Production**: `src/clean_room_agent/execute/implement.py:46-50` — always returns `success=True` or raises `ValueError`
- **Production**: `src/clean_room_agent/orchestrator/runner.py:1440` — `if not step_result.success`
- **Issue**: `execute_implement` never returns `success=False`. The mock `return_value = _make_step_result(False)` produces an impossible state. The `if not step_result.success` branch at runner.py:1440 is dead code.
- **Fix**: Change mock to `side_effect = ValueError("Parse failed")` and update assertions to test the actual failure path (exception handling at runner.py:1489).

### S10. `test_runner.py:69-76` — `_make_step_result(False)` produces impossible state — FIXED

- **Test**: `tests/orchestrator/test_runner.py:69-76`
- **Issue**: Helper creates `StepResult(success=False, ...)` which `execute_implement` and `execute_test_implement` can never produce. Any test using this helper with `success=False` as a `return_value` is testing an impossible state.
- **Fix**: Remove the `success=False` path from the helper or mark it clearly as "for testing dead-code paths only."

### S11. `runner.py:1440` — Dead code: `if not step_result.success` — FIXED

- **Production**: `src/clean_room_agent/orchestrator/runner.py:1440-1445`
- **Issue**: Since `execute_implement` always returns `success=True` or raises, this branch is unreachable. Left behind after the conversion.
- **Fix**: Remove the dead branch or convert to a defensive assertion (`assert step_result.success`).

### S12. `test_no_fallbacks.py:68-79` — `DICT_GET_ALLOWLIST` is dead code

- **Test**: `tests/test_no_fallbacks.py:68-79`
- **Issue**: Defines 9 allowlist entries but no test function references or enforces it. None of the 5 test functions scan for `.get()` usage.
- **Fix**: Either add a test that enforces the allowlist against `.get()` usage in production code, or delete the dead data.

### S13. `test_no_fallbacks.py:76-77` — Allowlist entries reference wrong files

- **Test**: `tests/test_no_fallbacks.py:76-77`
- **Issue**: Entries reference `retrieval/batch_judgment.py` for `.get("path")` and `.get("pair_id")`, but those calls are now in `scope_stage.py:276` and `similarity_stage.py:230`. `batch_judgment.py` has zero `.get()` calls.
- **Fix**: Update file references if the allowlist is kept; delete if S12 is resolved by removing it.

### S14. `test_no_fallbacks.py:78` — Allowlist entry for nonexistent `.get()` in prompts.py

- **Test**: `tests/test_no_fallbacks.py:78`
- **Issue**: Entry references `.get("retrieval")` in `execute/prompts.py`. The module has zero `.get()` calls and no config dict access.
- **Fix**: Delete the entry.

---

## FRAGILE Findings

### F1–F4. `test_environment.py` — No KeyError tests for required keys

- **Test**: `tests/test_environment.py` (missing tests)
- **Production**: `src/clean_room_agent/environment.py:83,84,88,89`
- **Issue**: After conversion to `dict[key]`, four access points will `KeyError` on missing keys: `config["testing"]`, `testing_config["test_command"]`, `config["environment"]`, `env_config["coding_style"]`. No test covers any of these failure modes.
- **Fix**: Add four tests asserting `KeyError`/`pytest.raises` for each missing key.

### F5–F7. `test_router.py` — No KeyError tests for `temperature`, temperature sub-keys, `overrides`

- **Test**: `tests/llm/test_router.py` (missing tests)
- **Production**: `src/clean_room_agent/llm/router.py:22,65,67-69`
- **Issue**: `models_config["overrides"]`, `models_config["temperature"]`, `temps["coding"]`, `temps["reasoning"]`, `temps["classifier"]` are all direct accesses. Tests exist for `context_window` and `provider` missing, but not for these three.
- **Fix**: Add `test_missing_overrides_raises`, `test_missing_temperature_raises`, `test_missing_temperature_subkeys_raises`.

### F8. `enrichment.py:135` — `public_api_surface` accessed via `dict[key]` but not validated — FIXED

- **Test**: `tests/llm/test_enrichment.py` (no happy-path test)
- **Production**: `src/clean_room_agent/llm/enrichment.py:135,156`
- **Issue**: `_REQUIRED_ENRICHMENT_FIELDS` has `("purpose", "module", "domain", "concepts")` but `parsed["public_api_surface"]` at line 156 is also a direct access. A response passing validation but missing this field crashes with a raw `KeyError`.
- **Fix**: Add `"public_api_surface"` to `_REQUIRED_ENRICHMENT_FIELDS`.

### F9. `precision_stage.py:199` — `cl["reason"]` crashes in malformed-response branch

- **Test**: `tests/retrieval/test_precision_stage.py` (no test covers this path)
- **Production**: `src/clean_room_agent/retrieval/precision_stage.py:199`
- **Issue**: When LLM returns a response with `name`/`file_path`/`start_line` but missing BOTH `detail_level` AND `reason`, the `elif "detail_level" not in cl` branch fires and `cl["reason"]` crashes. Previously `cl.get("reason", "")`.
- **Fix**: Either validate `"reason"` in this branch or use `.get("reason", "omitted")`.

### F10. `precision_stage.py:207-209` — Warning says "using empty" but next line crashes (**production bug**) — FIXED

- **Test**: `tests/retrieval/test_precision_stage.py` (all mocks provide `"reason"`)
- **Production**: `src/clean_room_agent/retrieval/precision_stage.py:207-209`
- **Issue**: Lines 207-208: `if "reason" not in cl: logger.warning("...missing 'reason'...—using empty")`. Line 209: `reason = cl["reason"]` — crashes with `KeyError`. The warning promises a graceful fallback that the code no longer performs.
- **Fix**: Either (a) restore `cl.get("reason", "")` to match the warning, or (b) remove the warning and let `KeyError` propagate as genuine fail-fast. The current state is contradictory.

### F11. `scope_stage.py:292` — `v["reason"]` KeyError on incomplete LLM verdict

- **Test**: `tests/retrieval/test_scope_stage.py` (all mocks include `"reason"`)
- **Production**: `src/clean_room_agent/retrieval/scope_stage.py:292`
- **Issue**: `sf.reason = v["reason"]` crashes if LLM returns `{"verdict": "relevant"}` without `"reason"`. Previously `v.get("reason", sf.reason)`.
- **Fix**: Validate in the batch judgment parsing or restore `.get()` with the existing reason as default.

### F12. `similarity_stage.py:241-254` — Three KeyErrors on incomplete LLM response

- **Test**: `tests/retrieval/test_similarity_stage.py` (all mocks provide all fields)
- **Production**: `src/clean_room_agent/retrieval/similarity_stage.py:241,243,244`
- **Issue**: `j["keep"]`, `j["group_label"]`, `j["reason"]` are all direct accesses. An LLM response missing any of these crashes instead of being treated as a malformed judgment.
- **Fix**: Validate required fields before access, or restore `.get()` with appropriate defaults for LLM response boundary parsing.

### F13. `test_pipeline.py` — No test for missing `affected_files` in plan artifact

- **Test**: `tests/retrieval/test_pipeline.py:274`
- **Production**: `src/clean_room_agent/retrieval/pipeline.py:255` — `plan_data["affected_files"]`
- **Issue**: Test provides `{"affected_files": ["src/main.py"]}`. No test covers a malformed plan artifact missing the key.
- **Fix**: Add test with plan artifact missing `affected_files`, asserting `KeyError`.

### F14. Multiple test files — `MagicMock` auto-satisfies `hasattr(llm, "flush")`

- **Tests**: `test_similarity_stage.py:29`, `test_task_analysis.py:173`, `test_precision_stage.py:24`, `test_scope_stage.py:183`, `test_context_assembly.py:469,510,645,684,756,789`
- **Production**: `_require_logged_client()` checks `hasattr(llm, "flush")`
- **Issue**: `MagicMock()` auto-creates any attribute on access, so `hasattr(mock, "flush")` is always `True`. Tests pass by accident of mock implementation, not by genuine contract compliance. If the guard were `isinstance(llm, LoggedLLMClient)`, all tests would break.
- **Fix**: Use `Mock(spec=LoggedLLMClient)` or explicitly set `mock_llm.flush = MagicMock()` to make the contract visible.

### F15. `test_loader.py:106-173` — Error-path TOMLs missing keys required by `dict[key]`

- **Test**: `tests/audit/test_loader.py:106-173` (5 error-path tests)
- **Production**: `src/clean_room_agent/audit/loader.py:51,57,63-68`
- **Issue**: Error-path TOML content is missing `budget_range`, `should_contain_files`, `must_not_contain`, `must_contain_information`, and `[routing_notes]`. Tests pass because the intended `ValueError` fires before the `dict[key]` accesses. If validation order changes, tests will `KeyError` instead.
- **Fix**: Add all required keys to error-path TOML content, even if the tests are testing other validation failures.

### F16. `test_p2_fixes.py:55-98` — Missing orchestrator config keys

- **Test**: `tests/test_p2_fixes.py:55-74,79-98` (`test_negative_max_diff_chars_raises`, `test_zero_max_diff_chars_raises`)
- **Production**: `src/clean_room_agent/orchestrator/runner.py:454-459`
- **Issue**: Orchestrator config dicts are missing `documentation_pass`, `scaffold_enabled`, `scaffold_compiler`, `scaffold_compiler_flags`. Tests pass because bounds-check `RuntimeError` fires before reaching these direct accesses. If validation order changes, tests break.
- **Fix**: Add `"documentation_pass": False, "scaffold_enabled": False, "scaffold_compiler": "gcc", "scaffold_compiler_flags": "-c -fsyntax-only -Wall"` to both test configs.

### F17. `test_library_indexing.py` — No test for `index_libraries()` with None/empty config

- **Test**: `tests/indexer/test_library_indexing.py` (all configs provide `library_sources`, `library_paths`)
- **Production**: `src/clean_room_agent/indexer/library_scanner.py:47-48` — `config["library_sources"]`, `config["library_paths"]`
- **Production caller**: `src/clean_room_agent/indexer/orchestrator.py:453` — passes `ic or {}` to `resolve_library_sources`
- **Issue**: `index_libraries(repo_path)` with no config or config missing `library_sources` will `KeyError`. The function signature allows `indexer_config: dict | None = None`. No test covers this.
- **Fix**: Add test for `KeyError` on empty config, or fix `index_libraries` to provide default config before passing to `resolve_library_sources`.

---

## Priority Matrix

### Must Fix (production crashes or wrong behavior)

| ID | Severity | Effort | Description | Status |
|---|---|---|---|---|
| B1 | Critical | 5 min | trace.py `call["thinking"]` — crashes on every non-thinking-model call | FIXED |
| B2 | Critical | 5 min | trace.py `call["error"]` — crashes on every successful LLM call | FIXED |
| B3 | High | 15 min | runner.py code step loop — marks failed patches as applied | FIXED |
| B4 | High | 15 min | runner.py test step loop — same | FIXED |
| F10 | High | 5 min | precision_stage.py:207-209 — warning/crash contradiction | FIXED |

### Should Fix (stale tests mislead developers)

| ID | Severity | Effort | Description | Status |
|---|---|---|---|---|
| S1-S4 | Medium | 15 min | Rename 4 tests with misleading names | |
| S5, F8 | Medium | 10 min | Add `public_api_surface` to `_REQUIRED_ENRICHMENT_FIELDS` | FIXED |
| S9-S11 | Medium | 15 min | Remove/rewrite dead-code tests around `success=False` | FIXED |
| S12-S14 | Low | 10 min | Clean up dead `DICT_GET_ALLOWLIST` | |

### Good to Have (lock in fail-fast contracts)

| ID | Severity | Effort | Description |
|---|---|---|---|
| F1-F7 | Low | 20 min | Add KeyError tests for environment.py, router.py required keys |
| F9,F11,F12 | Low | 15 min | Decide on LLM response boundary strategy (validate or `.get()`) |
| F14 | Low | 20 min | Replace `MagicMock()` with `Mock(spec=LoggedLLMClient)` |
| F15-F16 | Low | 10 min | Complete config dicts in error-path tests |

---

## Design Decision Required: LLM Response Boundary

Findings F9-F12 share a common pattern: LLM responses are unstructured dicts where any key can be absent. The `.get()` to `dict[key]` conversion applies fail-fast to these responses, but LLM responses are an **external system boundary** where malformed data is expected, not exceptional.

**Option A: Validate-then-access.** Add required-field validation before `dict[key]` access (like `_REQUIRED_ENRICHMENT_FIELDS`). Missing fields get a clear error message. Fail-fast principle preserved.

**Option B: Restore `.get()` at boundaries.** Treat LLM response parsing as an external boundary (like `tsconfig.json`). Use `.get()` with explicit defaults. Document in the allowlist.

**Option C: Batch judgment handles it.** The `run_batched_judgment` already has R2 default-deny logic. Extend it to validate required fields per stage before returning results.

**Recommendation**: Option A or C. The batch judgment layer is the natural validation point — it already filters malformed responses. Adding required-field validation there keeps `dict[key]` in stage code while ensuring clear errors.
