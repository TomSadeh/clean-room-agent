# Retrieval Audit Protocol

Systematic evaluation of retrieval quality. The protocol measures whether the
final context package contains what a competent engineer would need to complete
the task. It does NOT prescribe how the pipeline should find that information.

## Directory Layout

```
protocols/retrieval_audit/
  reference_tasks/     - TOML files defining tasks with known-correct context requirements
  findings/            - One TOML file per audit run with scores + findings
  README.md            - This file
```

## Running

```bash
# Run full audit suite
cra audit --repo .

# Run single reference task
cra audit --repo . --task RT-001

# Run with trace output
cra audit --repo . --trace
```

## Reference Task Schema

See any file in `reference_tasks/` for the TOML schema. Key fields:

- `must_contain_files` — task is impossible without these (hard failure)
- `should_contain_files` — task is harder without these (soft signal)
- `must_not_contain` — clearly irrelevant (budget waste)
- `must_contain_information` — what knowledge the execute model needs (content-level)
- `budget_range` — expected budget utilization as [min%, max%]

## Metrics

| Metric | Formula |
|--------|---------|
| Must-contain recall | present(must_contain_files) / total(must_contain_files) |
| Should-contain recall | present(should_contain_files) / total(should_contain_files) |
| Exclusion accuracy | absent(must_not_contain) / total(must_not_contain) |
| Budget utilization | tokens_used / budget_available |
| Parse success rate | successful_llm_parses / total_llm_calls |
| Task score | min(must_contain_recall, exclusion_accuracy) |
