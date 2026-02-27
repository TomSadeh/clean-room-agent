"""Deterministic domain/concept extraction from section titles and content."""

from __future__ import annotations

import re

from clean_room_agent.knowledge_base.models import RefSectionMeta


# Domain keywords — checked against title + first ~500 chars of content
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "pointers": ["pointer", "address", "dereference", "indirection", "->"],
    "memory_management": [
        "malloc", "calloc", "realloc", "free", "alloc", "memory",
        "heap", "stack", "storage duration",
    ],
    "io": [
        "printf", "scanf", "fprintf", "fscanf", "fopen", "fclose", "fread",
        "fwrite", "stdin", "stdout", "stderr", "input", "output", "stream",
        "file", "getchar", "putchar", "fgets", "fputs",
    ],
    "strings": [
        "string", "strlen", "strcpy", "strcat", "strcmp", "strncpy",
        "strncat", "strstr", "strtok", "char array", "null terminator",
    ],
    "control_flow": [
        "if-else", "switch", "while", "for", "do-while", "break",
        "continue", "goto", "loop", "branch", "jump",
    ],
    "data_types": [
        "int", "char", "float", "double", "long", "short", "unsigned",
        "signed", "enum", "typedef", "struct", "union", "size_t",
        "void", "type", "conversion", "cast",
    ],
    "preprocessor": [
        "preprocessor", "#define", "#include", "#ifdef", "#ifndef",
        "#if", "#else", "#endif", "macro", "#pragma",
    ],
    "concurrency": [
        "thread", "mutex", "lock", "semaphore", "atomic", "race condition",
        "deadlock", "concurrent", "pthread", "condition variable",
    ],
    "networking": [
        "socket", "bind", "listen", "accept", "connect", "send", "recv",
        "tcp", "udp", "ip", "port", "network", "client", "server",
    ],
    "systems": [
        "system call", "syscall", "process", "fork", "exec", "pipe",
        "signal", "interrupt", "kernel", "operating system", "os",
        "virtual memory", "page", "scheduling",
    ],
    "data_structures": [
        "linked list", "tree", "hash table", "queue", "stack",
        "array", "table", "binary", "node", "graph",
    ],
    "error_handling": [
        "error", "errno", "perror", "strerror", "exception",
        "assert", "abort", "exit",
    ],
    "bit_operations": [
        "bitwise", "bit", "shift", "mask", "xor",
        "complement", "twos complement",
    ],
}

# C standard library headers — for header detection
_HEADER_PATTERN = re.compile(r"<([\w.]+\.h)>")

# Common C standard library functions
_STDLIB_FUNCTIONS = {
    "malloc", "calloc", "realloc", "free", "printf", "scanf", "fprintf",
    "fscanf", "fopen", "fclose", "fread", "fwrite", "fgets", "fputs",
    "strlen", "strcpy", "strcat", "strcmp", "strncpy", "strncat", "strstr",
    "memcpy", "memmove", "memset", "memcmp", "getchar", "putchar",
    "puts", "gets", "atoi", "atof", "atol", "strtol", "strtod",
    "abs", "rand", "srand", "exit", "abort", "atexit", "system",
    "qsort", "bsearch", "isalpha", "isdigit", "isspace", "toupper",
    "tolower", "socket", "bind", "listen", "accept", "connect",
    "send", "recv", "fork", "exec", "pipe", "wait", "kill",
    "pthread_create", "pthread_join", "pthread_mutex_lock",
    "aligned_alloc", "free_sized", "free_aligned_sized",
}

# Function name pattern for cppreference-style content
_FUNC_PATTERN = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(")


def extract_metadata(title: str, content: str) -> RefSectionMeta:
    """Extract metadata from section title and content using keyword matching."""
    search_text = (title + " " + content[:500]).lower()

    # Domain: find best matching domain
    domain = _match_domain(search_text)

    # Concepts: collect all matching concept keywords
    concepts = _extract_concepts(search_text)

    # Header: find C headers mentioned
    header = _extract_headers(content)

    # Related functions: find stdlib functions mentioned
    related_functions = _extract_functions(content)

    # C standard: detect standard references
    c_standard = _detect_c_standard(content)

    return RefSectionMeta(
        domain=domain,
        concepts=",".join(concepts) if concepts else None,
        c_standard=c_standard,
        header=header,
        related_functions=",".join(related_functions) if related_functions else None,
    )


def _match_domain(text: str) -> str | None:
    """Find the best matching domain for the given text."""
    best_domain = None
    best_score = 0
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text)
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain


def _extract_concepts(text: str) -> list[str]:
    """Extract concept keywords present in the text."""
    concepts: set[str] = set()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text:
                concepts.add(kw)
    return sorted(concepts)


def _extract_headers(content: str) -> str | None:
    """Find C standard library headers mentioned in content."""
    headers = sorted(set(_HEADER_PATTERN.findall(content)))
    return ",".join(f"<{h}>" for h in headers) if headers else None


def _extract_functions(content: str) -> list[str]:
    """Find standard library function names in content."""
    found: set[str] = set()
    for match in _FUNC_PATTERN.finditer(content):
        name = match.group(1)
        if name in _STDLIB_FUNCTIONS:
            found.add(name)
    return sorted(found)


def _detect_c_standard(content: str) -> str | None:
    """Detect which C standard the content references."""
    text = content[:2000].lower()
    standards = []
    if "c23" in text or "c2x" in text:
        standards.append("C23")
    if "c17" in text or "c18" in text:
        standards.append("C17")
    if "c11" in text or "_generic" in text or "_atomic" in text:
        standards.append("C11")
    if "c99" in text or re.search(r"\brestrict\b", text) or re.search(r"\binline\b", text):
        standards.append("C99")
    if "c89" in text or "c90" in text or "ansi c" in text:
        standards.append("C89")
    return ",".join(standards) if standards else None
