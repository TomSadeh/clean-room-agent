# CLI Interface and Configuration

Command definitions, argument conventions, required inputs, config file format, and resolution order.

---

## 1. Commands

| Command | Primary argument | Phase | Requires LLM |
|---------|-----------------|-------|---------------|
| `cra init [repo-path]` | repo path (positional, default `.`) | Setup | No |
| `cra index [repo-path]` | repo path (positional, default `.`) | 1 | No |
| `cra enrich [repo-path]` | repo path (positional, default `.`) | 1 | Yes |
| `cra retrieve <task>` | task description (positional) | 2 | Yes |
| `cra plan <task>` | task description (positional) | 3 | Yes |
| `cra solve <task>` | task description (positional) | 3 | Yes |
| `cra train-plan` | (no positional) | 4 | Yes |
| `cra curate-data` | (no positional) | 4 | Yes |

---

## 2. Argument Conventions

- **Repo-focused commands** (`index`, `enrich`): repo path is the positional argument.
- **Task-focused commands** (`retrieve`, `plan`, `solve`): task description is the positional argument, `--repo <path>` is a named flag.
- **Training commands** (`train-plan`, `curate-data`): `--repo <path>` is a named flag. `curate-data` requires `--training-plan <artifact>`.
- **No `--model` or `--base-url` flags.** All LLM-using commands read model config from `.clean_room/config.toml`.

---

## 3. Required Inputs by Command

| Flag | `index` | `enrich` | `retrieve` | `plan` | `solve` | `train-plan` | `curate-data` |
|------|---------|----------|------------|--------|---------|--------------|---------------|
| `--stages` | -- | -- | optional* | optional* | -- | -- | -- |
| budget (see below) | -- | -- | optional* | -- | -- | -- | -- |
| `--plan` | -- | -- | optional | -- | optional | -- | -- |
| `--training-plan` | -- | -- | -- | -- | -- | -- | required |

\* "required" means: must be provided via CLI flag **or** config.toml. See resolution order (Section 5.2).

**Budget input**: `--context-window <int>` + `--reserved-tokens <int>` on `cra retrieve`, or config.toml `[budget]` section. `cra plan` and `cra solve` resolve budget entirely from config. Missing budget from all sources is a hard error.

**`--plan`** (on `cra solve`): bypasses the orchestrator entirely (single atomic implement pass from a pre-computed plan file).

---

## 4. Config File

### 4.1 Location and Format

`.clean_room/config.toml`, created by `cra init`.

```toml
[models]
provider = "ollama"                          # Required
# classifier = "qwen3:0.6b"                 # Optional: tier-0 binary classifier
coding = "qwen3:1.7b"                        # Required
reasoning = "qwen3:4b-instruct-2507"        # Required
base_url = "http://localhost:11434"          # Required
context_window = 32768                       # Required (int or per-role dict)
# max_tokens = 4096  # defaults to context_window // 8 if omitted
thinking = false                             # OFF: pipeline curates context, thinking is redundant

# Per-role context windows (when classifier has smaller window)
# [models.context_window]
# classifier = 8192
# coding = 32768
# reasoning = 32768

# Per-role max_tokens
# [models.max_tokens]
# classifier = 16
# coding = 4096
# reasoning = 4096

[models.overrides]
# scope = "qwen3:4b-scope-v1"

[models.teachers]     # Phase 4 only
# primary = "qwen3.5-plus"
# primary_base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
# primary_api_key_env = "DASHSCOPE_API_KEY"

# [models.temperature]
# coding = 0.0
# reasoning = 0.0
# classifier = 0.0

[budget]
# context_window: defaults to [models].context_window if omitted
reserved_tokens = 4096                       # Required

[stages]
default = "scope,precision"                  # Required

[testing]
test_command = "pytest tests/"               # Required
# lint_command = "ruff check src/"
# type_check_command = "mypy src/"
timeout = 120                                # Required

[orchestrator]
max_retries_per_step = 1                     # Required
# max_retries_per_test_step = 1  # defaults to max_retries_per_step if omitted
max_adjustment_rounds = 3                    # Required
git_workflow = true                          # Required
max_cumulative_diff_chars = 50000            # Required
documentation_pass = true                    # Optional (default: true)
# scaffold_enabled = false                   # Optional: C-only scaffold-then-implement
# scaffold_compiler = "gcc"                  # Optional: compiler for scaffold validation
# scaffold_compiler_flags = "-c -fsyntax-only -Wall"  # Optional: compiler flags

# [retrieval]
# max_deps = 30              # tier-2 dependency cap
# max_co_changes = 20        # tier-3 co-change cap
# max_metadata = 20          # tier-4 metadata cap
# max_keywords = 5           # keywords used for metadata search
# max_symbol_matches = 10    # LIKE symbol matches per name
# max_callees = 5            # callee connections per symbol
# max_callers = 5            # caller connections per symbol

# [indexer]
# max_file_size = 1048576    # skip files larger than this (bytes)
# co_change_max_files = 50   # skip commits touching more files
# co_change_min_count = 2    # minimum co-change count to keep
# max_commits = 500          # git log depth

[environment]
coding_style = "development"  # options: development, maintenance, prototyping
```

`cra init [repo-path]` creates a default config template. Users edit it to provide model, budget, and other settings.

### 4.2 Resolution Order

For every CLI input:

1. CLI flag present -> use it.
2. CLI flag absent, config.toml has the value -> use config value.
3. Neither -> **hard error** for required values.

**Code never has hardcoded defaults for required values.** Config is a convenience to avoid retyping the same flags. It is not a silent fallback -- it is an explicit value source checked at the CLI layer.

**Model config**: there are no CLI flags for model selection. Models are always read from config.toml. Missing `[models]` section -> hard error.

### 4.3 Config Loader

`config.py`: reads TOML, returns a flat dict. Missing file returns `None` (not an error -- config is optional for `cra index`). Missing `[models]` section when an LLM command runs is a hard error at the CLI layer.

### 4.4 Config Classification Policy

Every config field must be classified into one of three tiers. The classification determines how missing values are handled in code and whether the field appears uncommented in the default template.

| Tier | Missing behavior | Template | Code pattern |
|------|-----------------|----------|-------------|
| **Required** | Hard error (`raise RuntimeError`) | Uncommented with value | Direct access or explicit None check + raise |
| **Optional** | Documented default (named constant or derivation) | Commented with default shown | `resolve_*()` helper or explicit None check + log + fallback |
| **Supplementary** | Safe fallback, non-core | Commented or absent | `.get(key, default)` — only for non-core settings |

**Rules for adding new fields:**

1. Classify the field before writing any code. The classification goes in this table.
2. Required fields must be uncommented in `create_default_config()` and fail-fast when absent.
3. Optional fields must use a named constant or documented derivation (not a magic number in `.get()`). When the fallback fires, log the decision at INFO or higher.
4. No field may exist in an unclassified state. If a `.get(key, default)` exists in core logic without a classification entry here, it is a bug.

**Field classifications:**

| Section | Field | Tier | Default / derivation | Notes |
|---------|-------|------|---------------------|-------|
| `[models]` | `provider` | Required | — | |
| `[models]` | `classifier` | Optional | absent (binary judgment disabled) | When present, enables binary judgment in scope/similarity stages |
| `[models]` | `coding` | Required | — | |
| `[models]` | `reasoning` | Required | — | |
| `[models]` | `base_url` | Required | — | |
| `[models]` | `context_window` | Required | — | int or `{classifier=N, coding=N, reasoning=N}` |
| `[models]` | `max_tokens` | Optional | `context_window // 8` | int or `{classifier=N, coding=N, reasoning=N}` |
| `[models]` | `temperature` | Supplementary | `0.0` for all roles | |
| `[models]` | `overrides` | Supplementary | `{}` | Per-stage model overrides |
| `[models]` | `thinking` | Supplementary | `false` | OFF by default — pipeline curates context, thinking tokens are redundant retrieval. Enable for A/B testing only. |
| `[budget]` | `reserved_tokens` | Required | — | |
| `[stages]` | `default` | Required | — | Comma-separated stage names |
| `[testing]` | `test_command` | Required | — | |
| `[testing]` | `timeout` | Required | — | Seconds |
| `[testing]` | `lint_command` | Supplementary | absent | |
| `[testing]` | `type_check_command` | Supplementary | absent | |
| `[orchestrator]` | `max_retries_per_step` | Required | — | |
| `[orchestrator]` | `max_adjustment_rounds` | Required | — | |
| `[orchestrator]` | `git_workflow` | Required | — | |
| `[orchestrator]` | `max_cumulative_diff_chars` | Required | — | Positive integer |
| `[orchestrator]` | `max_retries_per_test_step` | Optional | `max_retries_per_step` | Logged fallback |
| `[orchestrator]` | `documentation_pass` | Optional | `true` | |
| `[orchestrator]` | `scaffold_enabled` | Optional | `false` | C-only: scaffold-then-implement |
| `[orchestrator]` | `scaffold_compiler` | Optional | `"gcc"` | Only read when scaffold_enabled=true; fail-fast if not on PATH |
| `[orchestrator]` | `scaffold_compiler_flags` | Optional | `"-c -fsyntax-only -Wall"` | Compiler flags for scaffold validation |
| `[retrieval]` | `max_deps` | Optional | `30` | Via `resolve_retrieval_param()` |
| `[retrieval]` | `max_co_changes` | Optional | `20` | Via `resolve_retrieval_param()` |
| `[retrieval]` | `max_metadata` | Optional | `20` | Via `resolve_retrieval_param()` |
| `[retrieval]` | `max_keywords` | Optional | `5` | Via `resolve_retrieval_param()` |
| `[retrieval]` | `max_symbol_matches` | Optional | `10` | Via `resolve_retrieval_param()` |
| `[retrieval]` | `max_callees` | Optional | `5` | Via `resolve_retrieval_param()` |
| `[retrieval]` | `max_callers` | Optional | `5` | Via `resolve_retrieval_param()` |
| `[retrieval]` | `max_candidate_pairs` | Optional | `50` | Via `resolve_retrieval_param()` |
| `[retrieval]` | `min_composite_score` | Optional | `0.3` | Via `resolve_retrieval_param()` |
| `[retrieval]` | `max_group_size` | Optional | `8` | Via `resolve_retrieval_param()` |
| `[environment]` | `coding_style` | Supplementary | `"development"` | Validated against `CODING_STYLES` |
