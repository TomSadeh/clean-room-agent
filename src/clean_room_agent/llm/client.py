"""LLM client for Ollama HTTP transport.

LLMClient is the raw transport layer.  LoggedLLMClient wraps it and records
every call for the traceability chain — use LoggedLLMClient in all production
code so that LLM I/O logging cannot be forgotten.
"""

import time
from dataclasses import dataclass

import httpx

from clean_room_agent.token_estimation import CHARS_PER_TOKEN_CONSERVATIVE


@dataclass
class ModelConfig:
    model: str
    base_url: str
    provider: str = "ollama"
    temperature: float = 0.0
    max_tokens: int = 4096
    context_window: int = 32768

    def __post_init__(self):
        if self.max_tokens >= self.context_window:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) must be < "
                f"context_window ({self.context_window})"
            )


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int


class LLMClient:
    """Thin transport layer for Ollama's /api/generate endpoint."""

    def __init__(self, config: ModelConfig):
        self.config = config
        if config.provider != "ollama":
            raise RuntimeError(
                f"LLM provider '{config.provider}' is not implemented in this MVP client."
            )
        self._http = httpx.Client(timeout=300.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __del__(self):
        try:
            self._http.close()
        except Exception:
            pass

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Send a completion request to Ollama. Fail-fast, no retry."""
        # R3: Budget-validate input before sending
        # Conservative estimate to avoid undercounting (matches batch sizing)
        input_tokens = (len(prompt) + (len(system) if system else 0)) // CHARS_PER_TOKEN_CONSERVATIVE
        available = self.config.context_window - self.config.max_tokens
        if input_tokens > available:
            raise ValueError(
                f"Input too large for model context window: ~{input_tokens} tokens "
                f"estimated, but only {available} tokens available "
                f"(context_window={self.config.context_window}, "
                f"max_tokens={self.config.max_tokens}). "
                f"Batch or pre-filter the input."
            )

        url = f"{self.config.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if system:
            payload["system"] = system

        start = time.monotonic()
        response = self._http.post(url, json=payload)
        response.raise_for_status()
        elapsed_ms = int((time.monotonic() - start) * 1000)

        data = response.json()
        if "response" not in data:
            raise RuntimeError(
                f"Ollama response missing 'response' field. Keys: {list(data.keys())}"
            )
        return LLMResponse(
            text=data["response"],
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_ms=elapsed_ms,
        )


class LoggedLLMClient:
    """LLM client that records all calls for the traceability chain.

    Wraps LLMClient and captures full I/O for every call.  Use this in all
    production code paths — retrieval stages, enrichment, Phase 3 execute
    stages — so that logging is automatic and cannot be forgotten.

    After a batch of calls, use ``flush()`` to retrieve and clear the
    accumulated call records, then write them to the raw DB.
    """

    def __init__(self, config: ModelConfig):
        self._client = LLMClient(config)
        self.calls: list[dict] = []

    @property
    def config(self) -> ModelConfig:
        return self._client.config

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Send a completion request and record the full I/O."""
        start = time.monotonic()
        response = self._client.complete(prompt, system=system)
        elapsed = int((time.monotonic() - start) * 1000)
        self.calls.append({
            "prompt": prompt,
            "system": system,
            "response": response.text,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "elapsed_ms": elapsed,
        })
        return response

    def flush(self) -> list[dict]:
        """Return and clear accumulated call records."""
        calls = self.calls
        self.calls = []
        return calls

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
