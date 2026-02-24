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
| `--stages` | -- | -- | required* | required* | required* | required* | required* |
| budget (see below) | -- | -- | required* | required* | required* | required* | required* |
| `--plan` | -- | -- | -- | -- | optional | -- | -- |
| `--training-plan` | -- | -- | -- | -- | -- | -- | required |
| `--meta-plan` | -- | -- | -- | -- | optional | -- | -- |
| `--max-attempts` | -- | -- | -- | -- | required* | -- | -- |
| `--max-refinement-loops` | -- | -- | -- | -- | required* | -- | -- |

\* "required" means: must be provided via CLI flag **or** config.toml. See resolution order (Section 5.2).

**Budget input**: either `--context-window <int>` + `--reserved-tokens <int>`, or `--budget-config <path>`. Mutually exclusive. Falls back to config.toml `[budget]` section if neither CLI form is provided. Missing budget from all sources is a hard error.

**`--plan` vs `--meta-plan`** (on `cra solve`): Mutually exclusive. `--plan` bypasses the orchestrator entirely (single atomic implement pass). `--meta-plan` skips only the meta-plan decomposition step; the orchestrator continues with part-plan, step implementation, and adjustment passes.

---

## 4. `--budget-config` File Format

TOML file:

```toml
context_window = 32768
reserved_tokens = 4096
```

No section header required. Just the two keys.

---

## 5. Config File

### 5.1 Location and Format

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

[budget]
context_window = 32768
reserved_tokens = 4096

[solve]
max_attempts = 3
max_refinement_loops = 2

[orchestrator]
max_parts = 10
max_steps_per_part = 15
max_adjustment_rounds = 3

[stages]
default = "scope,precision"
```

`cra init [repo-path]` creates a default config template. Users edit it to provide model, budget, and other settings.

### 5.2 Resolution Order

For every CLI input:

1. CLI flag present -> use it.
2. CLI flag absent, config.toml has the value -> use config value.
3. Neither -> **hard error** for required values.

**Code never has hardcoded defaults for required values.** Config is a convenience to avoid retyping the same flags. It is not a silent fallback -- it is an explicit value source checked at the CLI layer.

**Model config**: there are no CLI flags for model selection. Models are always read from config.toml. Missing `[models]` section -> hard error.

### 5.3 Config Loader

`config.py`: reads TOML, returns a flat dict. Missing file returns `None` (not an error -- config is optional for `cra index`). Missing `[models]` section when an LLM command runs is a hard error at the CLI layer.
