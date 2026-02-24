"""Direct Ollama test: Can Qwen3-4B-Thinking produce structured JSON plans?

Tests the thinking model with a simplified planning prompt and real code context.
The thinking model always outputs <think>...</think> before the actual response.
We need to strip the thinking block and parse the remainder as JSON.
"""

import json
import re
import sys
import time

import httpx

MODEL = "qwen3:4b-thinking-2507-q8_0"
BASE_URL = "http://localhost:11434"

# -- Simplified system prompt (much shorter than the full META_PLAN_SYSTEM) --

SYSTEM_PROMPT = """\
You are a task decomposition planner. Given code context and a task, decompose it into parts.

Output a JSON object:
{
  "task_summary": "...",
  "parts": [
    {
      "id": "p1",
      "description": "...",
      "affected_files": ["path/to/file.py"],
      "depends_on": []
    }
  ],
  "rationale": "..."
}

Rules:
- No code in plans, only describe changes
- Output ONLY valid JSON, no markdown, no extra text
"""

# -- Real task and context from maya-chat --

TASK = """\
Refactor context_retrieval_service.py to reduce complexity. The class has too many
private methods doing similar patterns (fetch data, format as ContextItem, return list).
Extract a common pattern and simplify.
"""

# Trimmed version of the actual file for context budget
CODE_CONTEXT = """\
# File: backend/services/context_retrieval_service.py (441 lines)

class ContextRetrievalService:
    def get_unified_context(self, request, session_id, student_id) -> RetrievedContext:
        # Main entry: checks request flags, calls private methods, combines results
        # Calls: _get_learning_progress, _get_current_exercise, _get_lesson_part_content,
        #        _get_part_exercises, _get_current_lesson_content, _get_file_contents
        # Returns RetrievedContext(items, total_tokens, retrieval_time_ms)
        ...

    def _get_learning_progress(self, student_id) -> List[ContextItem]:
        # Fetches progress, filters NOT_STARTED, maps status to Hebrew, returns ContextItems
        ...

    def _get_current_exercise(self, student_id) -> List[ContextItem]:
        # Gets current exercise, formats Hebrew description, returns ContextItem
        ...

    def _get_current_lesson_content(self, student_id) -> List[ContextItem]:
        # Gets current lesson, adds Hebrew header, returns ContextItem
        ...

    def _get_lesson_content(self, module, student_id) -> List[ContextItem]:
        # Gets lesson by module (fallback to exercise's topic), returns ContextItem
        ...

    def _get_lesson_part_content(self, module, part, total_parts) -> List[ContextItem]:
        # Gets specific lesson part or falls back to full lesson, returns ContextItem
        ...

    def _get_part_exercises(self, module, part) -> List[ContextItem]:
        # Gets exercises for a part, or returns "improvise" hint, returns ContextItems
        ...

    def _get_file_contents(self, file_paths) -> List[ContextItem]:
        # Reads student files, wraps in ContextItem with python code block
        ...

    def format_for_prompt(self, context: RetrievedContext) -> str:
        # Groups items by source_type, orders sections, joins with headers
        ...

# Related: ContextItem(source_type, content, topic_id=None, metadata={})
# Related: RetrievedContext(items, total_tokens, retrieval_time_ms)
"""


def strip_thinking(text: str) -> str:
    """Strip <think>...</think> block from model output.

    The thinking model always outputs thinking content first.
    The actual response follows after </think>.
    """
    # Find the end of the thinking block
    match = re.search(r"</think>\s*", text)
    if match:
        return text[match.end():]
    # No thinking block found — return as-is
    return text


def call_ollama(system: str, user: str, *, temperature: float = 0.6) -> dict:
    """Call Ollama and return raw response data."""
    return call_ollama_custom(system, user, temperature=temperature, num_predict=4096)


def call_ollama_custom(system: str, user: str, *, temperature: float = 0.6, num_predict: int = 4096) -> dict:
    """Call Ollama with custom parameters and return raw response data."""
    client = httpx.Client(timeout=600.0)
    try:
        start = time.monotonic()
        resp = client.post(
            f"{BASE_URL}/api/generate",
            json={
                "model": MODEL,
                "system": system,
                "prompt": user,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                },
            },
        )
        resp.raise_for_status()
        elapsed = time.monotonic() - start
        data = resp.json()
        data["_elapsed_s"] = round(elapsed, 1)
        return data
    finally:
        client.close()


def run_test(label, system, user, *, temperature, num_predict=4096, verbose=True):
    """Run a single test and return (parsed_json_or_None, raw_response)."""
    print(f"\n{'=' * 70}")
    print(f"[{label}] temp={temperature}, num_predict={num_predict}")

    data = call_ollama_custom(system, user, temperature=temperature, num_predict=num_predict)
    raw = data["response"]
    elapsed = data["_elapsed_s"]
    prompt_tok = data.get("prompt_eval_count", "?")
    completion_tok = data.get("eval_count", "?")
    print(f"  Latency: {elapsed}s | Prompt: {prompt_tok} tok | Completion: {completion_tok} tok")

    # Show thinking stats
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        print(f"  [Thinking block] ({len(thinking)} chars, ~{len(thinking)//4} tok)")
        if verbose:
            preview = thinking[:400] + ("..." if len(thinking) > 400 else "")
            for line in preview.split("\n"):
                print(f"    {line}")

    clean = strip_thinking(raw)
    print(f"\n  [Response after stripping] ({len(clean)} chars):")
    print(f"  ---")
    for line in clean.split("\n"):
        print(f"  {line}")
    print(f"  ---")

    # Try JSON parse
    json_text = clean.strip()
    if json_text.startswith("```"):
        lines = json_text.split("\n")
        json_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        parsed = json.loads(json_text)
        print(f"\n  [JSON PARSE: SUCCESS]")
        print(f"  task_summary: {parsed.get('task_summary', 'MISSING')}")
        parts = parsed.get("parts", [])
        print(f"  parts: {len(parts)}")
        for p in parts:
            print(f"    {p.get('id', '?')}: {p.get('description', '?')[:80]}")
            print(f"       files: {p.get('affected_files', [])} | depends_on: {p.get('depends_on', [])}")
        print(f"  rationale: {parsed.get('rationale', 'MISSING')[:120]}")
        return parsed, raw
    except json.JSONDecodeError as e:
        print(f"\n  [JSON PARSE: FAILED] {e}")
        print(f"  First 300 chars: {clean[:300]!r}")
        return None, raw


def main():
    print(f"Model: {MODEL}")
    print(f"Task: {TASK.strip()[:80]}...")

    user_prompt = f"# Code Context\n{CODE_CONTEXT}\n\n# Task\n{TASK}"

    only = sys.argv[1] if len(sys.argv) > 1 else "all"

    if only in ("all", "1"):
        run_test("Test 1: thinking temp=0.6 default budget",
                 SYSTEM_PROMPT, user_prompt, temperature=0.6, num_predict=4096)

    if only in ("all", "2"):
        run_test("Test 2: thinking temp=0.0 default budget",
                 SYSTEM_PROMPT, user_prompt, temperature=0.0, num_predict=4096)

    if only in ("all", "3"):
        run_test("Test 3: thinking temp=0.6 large budget",
                 SYSTEM_PROMPT, user_prompt, temperature=0.6, num_predict=16384)

    if only in ("all", "4"):
        run_test("Test 4: thinking temp=0.0 large budget",
                 SYSTEM_PROMPT, user_prompt, temperature=0.0, num_predict=16384)

    if only in ("all", "5"):
        # Test 5: Use the actual LLMClient with dynamic num_predict
        print(f"\n{'=' * 70}")
        print("[Test 5] Using clean-room-agent LLMClient (dynamic num_predict)")
        from clean_room_agent.llm.client import LLMClient, ModelConfig
        config = ModelConfig(
            model=MODEL,
            base_url=BASE_URL,
            temperature=0.6,
            max_tokens=4096,  # upstream planning budget (not the actual num_predict)
            context_window=32768,
        )
        client = LLMClient(config)
        start = time.monotonic()
        response = client.complete(user_prompt, system=SYSTEM_PROMPT)
        elapsed = round(time.monotonic() - start, 1)
        print(f"  Latency: {elapsed}s | Prompt: {response.prompt_tokens} tok | Completion: {response.completion_tokens} tok")
        if response.thinking:
            print(f"  [Thinking] ({len(response.thinking)} chars)")
        print(f"\n  [response.text] ({len(response.text)} chars):")
        print(f"  ---")
        for line in response.text.split("\n"):
            print(f"  {line}")
        print(f"  ---")

        try:
            json_text = response.text.strip()
            if json_text.startswith("```"):
                lines = json_text.split("\n")
                json_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            parsed = json.loads(json_text)
            print(f"\n  [JSON PARSE: SUCCESS]")
            print(f"  task_summary: {parsed.get('task_summary', 'MISSING')}")
            parts = parsed.get("parts", [])
            print(f"  parts: {len(parts)}")
            for p in parts:
                print(f"    {p.get('id', '?')}: {p.get('description', '?')[:80]}")
        except json.JSONDecodeError as e:
            print(f"\n  [JSON PARSE: FAILED] {e}")
        client.close()

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
