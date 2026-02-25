# Corpus Cloning Protocol -- Design Record

**Date:** 2026-02-26
**Status:** Decided

## Problem Statement

We need a general-purpose protocol for building a local repository corpus that serves three purposes: (1) LoRA training data extraction from commit histories (fail-fast coding style, LoRA-about-LoRAs), (2) curated coding context for novel/niche patterns a 3-4B model wouldn't know from pretraining, and (3) potentially documentation style training from gold-tier repos. The protocol must be secure against unwanted files, prompt injection, hidden text, and supply chain risks, and must be reusable for any future repo additions beyond the initial 45.

## Decision

Build a Python script (`clone_corpus.py`) + TOML config (`corpus_manifest.toml`) that implements a secure, tiered corpus cloning protocol with a 5-stage security pipeline including LLM verification.

## Architecture

### Config File: `corpus_manifest.toml`

Each repo entry defines:

```toml
[repos.structlog]
url = "https://github.com/hynek/structlog.git"
commit = "abc123def456"  # pinned at review time
clone_strategy = "full"  # "full" | "shallow"
purposes = ["fail-fast-training", "documentation-exemplar", "curated-context"]
doc_quality = 5.0
priority_score = 95  # combined: doc_quality + purpose_relevance + inverse_size
source_review = "fail_fast_research.md"
```

**Global config section:**

```toml
[config]
corpus_root = "../training-corpus"
extension_allowlist = [
    ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx",
    ".md", ".rst", ".txt",
    ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini",
    ".sh",
]
max_file_size_kb = 500
shallow_depth = 1
gemini_model = "gemini-2.0-flash"
sample_rate_clean_files = 0.05  # 5% random sample of unflagged files sent to LLM
```

### Directory Structure

```
../training-corpus/
    fail-fast/
        structlog/
        beartype/
        attrs/
        ...
    lora-training/
        unsloth/
        pydriller/
        ...
    manifest.json          # auto-generated corpus-wide manifest
    corpus_snapshot.tar.gz  # frozen canonical version after vetting
```

Each repo lives in one physical directory under its primary purpose. The manifest records all purposes per repo.

### Clone Pipeline (per repo)

#### Stage 1: Pre-Clone Vetting

- Verify repo URL is in the allowlist (corpus_manifest.toml).
- For new additions not yet reviewed: fetch repo metadata via GitHub API (size, file count, star count, last update) and log for human review before proceeding.

#### Stage 2: Secure Clone

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone \
    --no-checkout \
    --no-hooks \
    --no-recurse-submodules \
    <url> <target_dir>
```

For shallow clones, add `--depth=1`.
For full clones (training data repos), no depth limit.

Pin to reviewed commit:
```bash
cd <target_dir>
git checkout <pinned_commit_hash>
```

#### Stage 3: Pre-Checkout Security Scan

Before materializing ANY files, inspect the git tree:

1. **Submodule check:** `git ls-tree -r HEAD | grep 160000` — any gitlink entries mean submodules. Flag and halt.
2. **Gitattributes check:** `git show HEAD:.gitattributes` — scan for `filter=` directives that execute commands. Flag and halt.
3. **Hook check:** `git ls-tree -r HEAD -- .githooks/ .husky/` — log presence of hook directories (they won't execute due to `--no-hooks`, but document them).
4. **File extension audit:** `git ls-tree -r --name-only HEAD` — count files by extension. Log any extensions not on allowlist. These will NOT be checked out.
5. **File size audit:** `git ls-tree -r -l HEAD` — flag any files >500KB.

#### Stage 4: Sparse Checkout

Only materialize allowlisted file extensions:

```bash
git sparse-checkout init --cone
git sparse-checkout set --no-cone '*.py' '*.pyi' '*.ts' '*.tsx' '*.js' '*.jsx' '*.md' '*.rst' '*.txt' '*.toml' '*.yaml' '*.yml' '*.json' '*.cfg' '*.ini' '*.sh'
git checkout <pinned_commit_hash>
```

#### Stage 5: Post-Checkout Security Scans

On the materialized files:

1. **Size filter:** Flag any file >500KB (should be caught in Stage 3, but verify).

2. **Unicode scan:** Scan all text files for:
   - Zero-width characters (U+200B, U+200C, U+200D, U+FEFF, U+00AD)
   - RTL/LTR override characters (U+202A-U+202E, U+2066-U+2069)
   - Homoglyph characters (Cyrillic/Greek lookalikes for Latin chars)
   - Other invisible formatting (U+2060-U+2064, U+180E)
   Flag file + line number + character.

3. **Injection pattern scan:** Scan comments, docstrings, and string literals for patterns:
   - `ignore previous instructions`
   - `ignore all prior`
   - `you are now`
   - `system prompt`
   - `<\|im_start\|>system`
   - `[INST]`, `<<SYS>>`
   - `<system>`, `</system>`
   - `IMPORTANT:.*override`
   - `disregard.*above`
   - Any ChatML/instruction-format tokens embedded in source
   Flag file + line number + matched pattern.

4. **LLM verification pass (Gemini API):**
   - Send ALL files flagged by scans 1-3 to Gemini with assessment prompt.
   - Send a random sample (5% default) of UNFLAGGED files as a canary check.
   - Prompt asks Gemini to assess for: prompt injection attempts, hidden instructions in comments/docstrings, obfuscated code, suspicious encoded content, anything designed to manipulate an LLM trained on this code.
   - Log Gemini's assessment (safe/suspicious + reasoning) per file in manifest.

5. **All flags logged to manifest.** Human reviews flagged files before the repo is marked "vetted."

### Post-Clone Processing

Auto-generate manifest entry per repo:

```json
{
    "repo": "structlog",
    "url": "https://github.com/hynek/structlog.git",
    "commit": "abc123def456",
    "clone_strategy": "full",
    "purposes": ["fail-fast-training", "documentation-exemplar", "curated-context"],
    "doc_quality": 5.0,
    "source_review": "fail_fast_research.md",
    "stats": {
        "files_checked_out": 47,
        "files_excluded": 3,
        "total_commits": 1842,
        "size_on_disk_mb": 12.4,
        "languages": {"python": 42, "markdown": 3, "toml": 2}
    },
    "security": {
        "submodules": false,
        "git_filters": false,
        "hooks_present": false,
        "files_flagged": 0,
        "llm_verified_count": 3,
        "llm_suspicious_count": 0,
        "status": "vetted"
    }
}
```

### Snapshot

After all repos cloned and vetted:

```bash
tar -czf corpus_snapshot.tar.gz -C ../training-corpus .
```

This is the canonical frozen corpus. If any clone is corrupted or modified, restore from snapshot.

### Priority Order

Combined score: `(doc_quality * 10) + (purpose_count * 5) + (100 / size_estimate_mb)`.

**First 3 (protocol validation):** structlog, attrs, beartype — small, high-quality, multi-purpose.

**Then by descending priority score**, with training-data repos before context-only repos at equal score.

## Rationale

- **Tiered cloning** matches dual-use needs without wasting disk on full history for context-only repos.
- **Allowlist-not-blocklist** for file extensions ensures no unknown file types reach disk.
- **5-stage security pipeline** provides defense in depth: pre-checkout tree inspection catches structural risks, sparse checkout prevents unwanted files from materializing, post-checkout scans catch content-level risks, LLM verification catches what pattern matching misses.
- **`--no-checkout --no-hooks --no-recurse-submodules` + `GIT_LFS_SKIP_SMUDGE=1`** ensures nothing executes and nothing external is pulled during clone.
- **Commit pinning** freezes the corpus at the reviewed state. Updates are deliberate.
- **Manifest** makes the corpus queryable, reproducible, and auditable.
- **Gemini API for LLM verification** is cheap (only fires on flagged files + small random sample) and provides confidence that automated scans didn't miss adversarial content.

## Alternatives Considered

| Alternative | Why rejected |
|-------------|--------------|
| Full clone all repos | Disk/time overkill for context-only repos |
| Shallow clone all repos | Destroys commit history needed for LoRA training |
| Blobless clone (`--filter=blob:none`) | Unproven with PyDriller extraction tooling |
| Flat directory organization | No signal about purpose at 45+ repo scale |
| Symlinks for dual-use repos | Fragile on Windows |
| Store inside clean-room-agent repo | Risk of `git clean` wiping corpus |
| Store in `~/.clean_room/` runtime dir | Lifecycle mismatch with runtime state |
| Shell script automation | Too primitive for security scanning |
| `cra corpus` CLI integration | Premature; build-time tool not runtime feature |
| Extension blocklist | Blocklists always have gaps |
| Git signature verification | Most repos don't sign; low coverage for high effort |
| LLM review of entire corpus | Expensive; targeted approach (flags + sample) is sufficient |
| Diff-based injection scanning at clone time | Belongs in extraction phase, not clone phase |

## Open Questions

1. **Commit hash pinning logistics** -- Pin now via GitHub API, or at first clone time? Recommend: pin at clone time and record in manifest.
2. **Documentation LoRA viability** -- When/how to validate that gold-tier docs improve model output? Recommend: defer to Phase 4 experimentation.
3. **Disk budget** -- Estimated 5-15GB for full corpus. Need to verify after first 3 clones.
4. **Injection pattern list maintenance** -- Starting set defined above. How to update? Recommend: patterns live in the TOML config, reviewed on each corpus expansion.
5. **Extension allowlist gaps** -- `.pyi`, `.tsx`, `.jsx`, `.yml` included. Missing anything for future language expansion? Recommend: review when adding non-Python/TS repos.

## Validation Criteria

- **Success:** 45+ repos cloned, each with manifest entry showing: commit hash, clone strategy, purposes, file/commit counts, security scan status. No binaries on disk. No unvetted code executed. No LLM-flagged files unreviewed.
- **Test:** Clone top 3 repos (structlog, beartype, attrs). Verify manifest generation, security scan output, sparse checkout correctness, Gemini integration.
- **Failure signal:** Security scan misses a binary on disk. Sparse checkout excludes needed files. Manifest incomplete. Gemini API integration fails silently. Any of these means protocol revision before scaling.
