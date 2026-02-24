"""Task analysis: deterministic extraction + LLM intent enrichment."""

import logging
import re

from clean_room_agent.llm.client import LLMClient
from clean_room_agent.query.api import KnowledgeBase
from clean_room_agent.retrieval.dataclasses import TaskQuery

logger = logging.getLogger(__name__)

MAX_SYMBOL_MATCHES = 10

KNOWN_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because", "but",
    "and", "or", "if", "while", "about", "up", "it", "its", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "they", "them", "what", "which", "who", "whom",
})

TASK_ANALYSIS_SYSTEM = (
    "You are Jane, a code task analyzer. Given a task description and extracted signals, "
    "produce a concise 1-2 sentence intent summary that captures the core goal. "
    "Focus on what needs to change and why. Respond with ONLY the summary text, "
    "no formatting or labels."
)

# Regex patterns
_FILE_PATH_RE = re.compile(
    r'(?:^|[\s\'"`({,])([a-zA-Z0-9_./-]+(?:'
    + "|".join(re.escape(ext) for ext in KNOWN_EXTENSIONS)
    + r'))(?:[\s\'"`)},:;]|$)',
    re.MULTILINE,
)
_CAMEL_CASE_RE = re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b')
_SNAKE_CASE_RE = re.compile(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b')
_DOTTED_NAME_RE = re.compile(r'\b([A-Za-z]\w+\.[a-z_]\w+)\b')
_ERROR_PATTERN_RE = re.compile(
    r'^.*(?:Error:|Exception:|Traceback|at line)\s*.*$',
    re.MULTILINE | re.IGNORECASE,
)

# Task type heuristics: first match wins. Ordered by specificity:
# bug_fix before feature (both may match "fix the new feature" — bug intent takes priority).
_TASK_TYPE_RULES = [
    ({"fix", "bug", "broken", "crash", "error", "issue", "wrong", "fail"}, "bug_fix"),
    ({"add", "implement", "create", "new", "feature", "build", "introduce"}, "feature"),
    ({"refactor", "rename", "restructure", "reorganize", "move", "extract", "clean"}, "refactor"),
    ({"test", "spec", "coverage", "assert", "unittest", "pytest"}, "test"),
    ({"doc", "docs", "document", "readme", "comment", "docstring"}, "docs"),
]


def extract_task_signals(raw_task: str) -> dict:
    """Extract deterministic signals from a raw task description.

    Returns dict with keys: files, symbols, error_patterns, keywords, task_type
    """
    files = _FILE_PATH_RE.findall(raw_task)

    symbols = set()
    symbols.update(_CAMEL_CASE_RE.findall(raw_task))
    symbols.update(_SNAKE_CASE_RE.findall(raw_task))
    symbols.update(_DOTTED_NAME_RE.findall(raw_task))
    symbols_list = sorted(symbols)

    error_patterns = _ERROR_PATTERN_RE.findall(raw_task)

    # Keywords: significant words after removing stop words and extracted items
    extracted_lower = {item.lower() for item in (set(files) | symbols | set(error_patterns))}
    words = re.findall(r'\b[a-zA-Z_]\w*\b', raw_task.lower())
    keywords = [
        w for w in dict.fromkeys(words)  # dedupe preserving order
        if w not in STOP_WORDS and w not in extracted_lower and len(w) > 2
    ]

    # Task type heuristic
    task_words = set(re.findall(r'\b[a-zA-Z_]\w*\b', raw_task.lower()))
    task_type = "unknown"
    for keyword_set, ttype in _TASK_TYPE_RULES:
        if task_words & keyword_set:
            task_type = ttype
            break

    return {
        "files": files,
        "symbols": symbols_list,
        "error_patterns": error_patterns,
        "keywords": keywords,
        "task_type": task_type,
    }


def resolve_seeds(
    signals: dict,
    kb: KnowledgeBase,
    repo_id: int,
    *,
    max_symbol_matches: int = MAX_SYMBOL_MATCHES,
) -> tuple[list[int], list[int]]:
    """Resolve extracted signals to DB IDs.

    Returns (file_ids, symbol_ids). Logs unresolved at DEBUG.
    """
    file_ids = []
    for path in signals.get("files", []):
        f = kb.get_file_by_path(repo_id, path)
        if f:
            file_ids.append(f.id)
        else:
            logger.debug("Unresolved file path: %s", path)

    symbol_ids = []
    for name in signals.get("symbols", []):
        matches = kb.search_symbols_by_name(repo_id, name)
        if matches:
            # R6: prefer exact matches; for LIKE results, order by name length
            # (shorter = more specific match) before capping
            exact = [s for s in matches if s.name == name]
            if exact:
                selected = exact
            else:
                selected = sorted(matches, key=lambda s: len(s.name))
            symbol_ids.extend(s.id for s in selected[:max_symbol_matches])
        else:
            logger.debug("Unresolved symbol: %s", name)

    return file_ids, symbol_ids


def enrich_task_intent(
    raw_task: str,
    signals: dict,
    llm: LLMClient,
    repo_file_tree: str = "",
    environment_brief: str = "",
) -> str:
    """Use LLM to produce an intent summary from the task and extracted signals.

    When repo_file_tree and environment_brief are provided, this becomes the
    bird's-eye-view prompt: the model sees the whole repo layout alongside the
    task, enabling strategic decisions about where to look.
    """
    signal_text = (
        f"Extracted files: {signals.get('files', [])}\n"
        f"Extracted symbols: {signals.get('symbols', [])}\n"
        f"Task type: {signals.get('task_type', 'unknown')}\n"
        f"Keywords: {signals.get('keywords', [])}"
    )

    parts = []
    if environment_brief:
        parts.append(environment_brief)
    parts.append(f"Task: {raw_task}")
    parts.append(signal_text)
    if repo_file_tree:
        parts.append(f"<repo_structure>\n{repo_file_tree}\n</repo_structure>")
    prompt = "\n\n".join(parts)

    try:
        response = llm.complete(prompt, system=TASK_ANALYSIS_SYSTEM)
    except ValueError as e:
        raise ValueError(
            f"Task intent enrichment prompt exceeds model capacity. "
            f"Reduce task description or extracted signals. Original: {e}"
        ) from e
    return response.text.strip()


def analyze_task(
    raw_task: str,
    task_id: str,
    mode: str,
    kb: KnowledgeBase,
    repo_id: int,
    llm: LLMClient,
    repo_file_tree: str = "",
    environment_brief: str = "",
    retrieval_params: dict | None = None,
) -> TaskQuery:
    """Full task analysis: extract signals, resolve seeds, enrich intent."""
    signals = extract_task_signals(raw_task)
    rp = retrieval_params or {}
    file_ids, symbol_ids = resolve_seeds(
        signals, kb, repo_id,
        max_symbol_matches=rp.get("max_symbol_matches", MAX_SYMBOL_MATCHES),
    )
    intent = enrich_task_intent(
        raw_task, signals, llm,
        repo_file_tree=repo_file_tree,
        environment_brief=environment_brief,
    )

    return TaskQuery(
        raw_task=raw_task,
        task_id=task_id,
        mode=mode,
        repo_id=repo_id,
        mentioned_files=signals["files"],
        mentioned_symbols=signals["symbols"],
        keywords=signals["keywords"],
        error_patterns=signals["error_patterns"],
        task_type=signals["task_type"],
        intent_summary=intent,
        seed_file_ids=file_ids,
        seed_symbol_ids=symbol_ids,
    )
