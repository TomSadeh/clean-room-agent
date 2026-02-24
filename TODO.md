# TODO

Remaining items from the Phase 1 code review. Candidates for Phase 2 refactoring.

## Parser Cleanup

- **Unify `_classify_comment` logic** — `python_parser.py` uses `^#\s*TODO\b` (anchored, strips markers first) while `ts_js_parser.py` uses `\bTODO\b` (matches anywhere, receives pre-stripped text). The divergence is unnecessary; extract a shared helper.

- **Unify `_find_enclosing_symbol` signatures** — `python_parser.py` takes `(node, list[tuple])` while `ts_js_parser.py` takes `(int, list[ExtractedSymbol])`. Both find the innermost enclosing symbol by line range. Extract to a shared utility with a consistent interface.

## Minor Design Notes

- **Enrichment skip logic assumes `file_id` stability** — `enrich_repository` skips files already in `raw.enrichment_outputs` by `file_id`, but `file_id` values are not stable across curated DB rebuilds. If curated DB is deleted and re-indexed, previously-enriched files get new IDs and will be re-enriched (wasted LLM calls, no data loss). Consider checking by file path instead, or document this as accepted behavior.
