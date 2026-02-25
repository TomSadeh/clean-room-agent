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
provider = "ollama"                          # "ollama", "openai_compat", or "remote_api"
coding = "qwen2.5-coder:3b-instruct"
reasoning = "qwen3:4b-instruct-2507"
base_url = "http://localhost:11434"

[models.overrides]
# scope = "qwen3:4b-scope-v1"

[models.teachers]     # Phase 4 only
# primary = "qwen3.5-plus"
# primary_base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
# primary_api_key_env = "DASHSCOPE_API_KEY"

# [models.temperature]
# coding = 0.0
# reasoning = 0.0

[budget]
context_window = 32768
reserved_tokens = 4096

[orchestrator]
max_retries_per_step = 1
max_adjustment_rounds = 3
# max_cumulative_diff_chars = 50000

[stages]
default = "scope,precision"

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

[testing]
test_command = "pytest tests/"
# lint_command = "ruff check src/"
# type_check_command = "mypy src/"
# timeout = 120

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
