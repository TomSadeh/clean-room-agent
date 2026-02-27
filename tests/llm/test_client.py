"""Tests for llm/client.py."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from clean_room_agent.llm.client import (
    EnvironmentLLMClient, LLMClient, LLMResponse, LoggedLLMClient, ModelConfig,
    strip_thinking,
)


class TestModelConfig:
    def test_defaults(self):
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.provider == "ollama"

    def test_custom(self):
        config = ModelConfig(
            model="qwen3:4b", base_url="http://localhost:11434",
            temperature=0.5, max_tokens=2048,
        )
        assert config.temperature == 0.5
        assert config.max_tokens == 2048


class TestLLMResponse:
    def test_fields(self):
        resp = LLMResponse(text="hello", thinking=None, prompt_tokens=10, completion_tokens=5, latency_ms=100)
        assert resp.text == "hello"
        assert resp.prompt_tokens == 10

    def test_tokens_can_be_none(self):
        resp = LLMResponse(text="hello", thinking=None, prompt_tokens=None, completion_tokens=None, latency_ms=50)
        assert resp.prompt_tokens is None
        assert resp.completion_tokens is None


class TestStripThinking:
    def test_strips_thinking_block(self):
        raw = "<think>I need to plan this carefully.</think>\n{\"result\": true}"
        clean, thinking = strip_thinking(raw)
        assert clean == '{"result": true}'
        assert thinking == "I need to plan this carefully."

    def test_no_thinking_block(self):
        raw = '{"result": true}'
        clean, thinking = strip_thinking(raw)
        assert clean == '{"result": true}'
        assert thinking is None

    def test_multiline_thinking(self):
        raw = "<think>\nLine 1\nLine 2\nLine 3\n</think>\nclean output"
        clean, thinking = strip_thinking(raw)
        assert clean == "clean output"
        assert "Line 1" in thinking
        assert "Line 3" in thinking

    def test_whitespace_after_close_tag_stripped(self):
        raw = "<think>thought</think>   \n  response"
        clean, thinking = strip_thinking(raw)
        assert clean == "response"


class TestLLMClient:
    def test_instantiation(self):
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        client = LLMClient(config)
        assert client.config.model == "qwen3:4b"
        client.close()

    def test_non_ollama_provider_fails_fast(self):
        config = ModelConfig(
            model="qwen3:4b",
            base_url="http://localhost:11434",
            provider="openai_compat",
        )
        with pytest.raises(RuntimeError, match="not implemented"):
            LLMClient(config)

    def test_complete_sends_correct_payload(self):
        config = ModelConfig(
            model="qwen3:4b", base_url="http://localhost:11434",
            temperature=0.3, max_tokens=1024, context_window=32768,
        )
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Hello world",
            "prompt_eval_count": 15,
            "eval_count": 8,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._http, "post", return_value=mock_response) as mock_post:
            result = client.complete("test prompt", system="Be helpful")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
        assert payload["model"] == "qwen3:4b"
        assert payload["prompt"] == "test prompt"
        assert payload["system"] == "Be helpful"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.3
        # num_predict is dynamic: context_window - input_estimate - margin
        # "test prompt" + "Be helpful" = 21 chars -> ~7 tokens at chars/3
        # num_predict = 32768 - 7 - 256 = 32505
        assert payload["options"]["num_predict"] > 1024  # not the fixed max_tokens
        assert payload["options"]["num_predict"] < config.context_window

        assert result.text == "Hello world"
        assert result.thinking is None
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 8
        assert result.latency_ms >= 0
        client.close()

    def test_complete_strips_thinking_from_response(self):
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "<think>Let me reason about this.</think>\n{\"answer\": 42}",
            "prompt_eval_count": 10,
            "eval_count": 30,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._http, "post", return_value=mock_response):
            result = client.complete("test")

        assert result.text == '{"answer": 42}'
        assert result.thinking == "Let me reason about this."
        client.close()

    def test_complete_without_system_omits_key(self):
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._http, "post", return_value=mock_response) as mock_post:
            client.complete("test")

        payload = mock_post.call_args.kwargs["json"] if "json" in mock_post.call_args.kwargs else mock_post.call_args[1]["json"]
        assert "system" not in payload
        client.close()

    def test_complete_missing_response_field_raises(self):
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {"model": "qwen3:4b", "done": True}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._http, "post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="missing 'response' field"):
                client.complete("test")
        client.close()

    def test_complete_tokens_none_when_missing(self):
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._http, "post", return_value=mock_response):
            result = client.complete("test")

        assert result.prompt_tokens is None
        assert result.completion_tokens is None
        client.close()

    def test_context_manager(self):
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        with LLMClient(config) as client:
            assert client.config.model == "qwen3:4b"
        # After exiting, the underlying httpx client should be closed
        assert client._http.is_closed

    def test_close_idempotent(self):
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        client = LLMClient(config)
        client.close()
        client.close()  # should not raise

    def test_oversized_input_raises(self):
        """R3b: input exceeding context_window - max_tokens should raise ValueError."""
        config = ModelConfig(
            model="qwen3:4b", base_url="http://localhost:11434",
            max_tokens=1024, context_window=2048,
        )
        client = LLMClient(config)
        # available = 2048 - 1024 = 1024 tokens = ~4096 chars
        big_prompt = "x" * 5000  # ~1250 tokens > 1024 available
        with pytest.raises(ValueError, match="Input too large"):
            client.complete(big_prompt)
        client.close()

    def test_within_limit_succeeds(self):
        """R3b: input within budget should proceed normally."""
        config = ModelConfig(
            model="qwen3:4b", base_url="http://localhost:11434",
            max_tokens=1024, context_window=32768,
        )
        client = LLMClient(config)
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._http, "post", return_value=mock_response):
            result = client.complete("short prompt")
        assert result.text == "ok"
        client.close()

    def test_http_status_error_propagates(self):
        """HTTP 500 error propagates as httpx.HTTPStatusError."""
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=MagicMock(status_code=500),
        )

        with patch.object(client._http, "post", return_value=mock_response):
            with pytest.raises(httpx.HTTPStatusError, match="Server Error"):
                client.complete("test")
        client.close()

    def test_connect_error_propagates(self):
        """Connection refused propagates as httpx.ConnectError."""
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        client = LLMClient(config)

        with patch.object(client._http, "post", side_effect=httpx.ConnectError("Connection refused")):
            with pytest.raises(httpx.ConnectError, match="Connection refused"):
                client.complete("test")
        client.close()

    def test_timeout_error_propagates(self):
        """Timeout propagates as httpx.TimeoutException."""
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        client = LLMClient(config)

        with patch.object(client._http, "post", side_effect=httpx.TimeoutException("Timed out")):
            with pytest.raises(httpx.TimeoutException, match="Timed out"):
                client.complete("test")
        client.close()

    def test_oversized_system_prompt_raises(self):
        """R3: large system prompt alone exceeding budget should raise ValueError."""
        config = ModelConfig(
            model="qwen3:4b", base_url="http://localhost:11434",
            max_tokens=1024, context_window=2048,
        )
        client = LLMClient(config)
        # available = 2048 - 1024 = 1024 tokens. System alone ~1667 tokens > 1024.
        big_system = "x" * 5000
        with pytest.raises(ValueError, match="Input too large"):
            client.complete("short", system=big_system)
        client.close()

    def test_max_tokens_gte_context_window_raises(self):
        """ModelConfig rejects max_tokens >= context_window."""
        with pytest.raises(ValueError, match="max_tokens.*must be <"):
            ModelConfig(
                model="qwen3:4b", base_url="http://localhost:11434",
                max_tokens=32768, context_window=32768,
            )


class TestLoggedLLMClient:
    """T20: LoggedLLMClient records all calls for the traceability chain."""

    def _make_logged_client(self):
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        return LoggedLLMClient(config)

    def _mock_http_response(self, client, text="ok", prompt_tokens=10, completion_tokens=5):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": text,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
        }
        mock_response.raise_for_status = MagicMock()
        return patch.object(client._client._http, "post", return_value=mock_response)

    def test_complete_records_call(self):
        client = self._make_logged_client()
        with self._mock_http_response(client, "hello", 15, 8):
            result = client.complete("test prompt", system="be helpful")

        assert result.text == "hello"
        assert len(client.calls) == 1
        call = client.calls[0]
        assert call["prompt"] == "test prompt"
        assert call["system"] == "be helpful"
        assert call["response"] == "hello"
        assert call["prompt_tokens"] == 15
        assert call["completion_tokens"] == 8
        assert call["elapsed_ms"] >= 0
        client.close()

    def test_flush_returns_and_clears(self):
        client = self._make_logged_client()
        with self._mock_http_response(client):
            client.complete("call 1")
            client.complete("call 2")

        assert len(client.calls) == 2
        flushed = client.flush()
        assert len(flushed) == 2
        assert len(client.calls) == 0
        assert flushed[0]["prompt"] == "call 1"
        assert flushed[1]["prompt"] == "call 2"
        client.close()

    def test_multiple_flushes(self):
        client = self._make_logged_client()
        with self._mock_http_response(client):
            client.complete("batch 1")
        first = client.flush()
        assert len(first) == 1

        with self._mock_http_response(client):
            client.complete("batch 2")
        second = client.flush()
        assert len(second) == 1
        assert second[0]["prompt"] == "batch 2"
        client.close()

    def test_context_manager(self):
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        with LoggedLLMClient(config) as client:
            assert client.config.model == "qwen3:4b"
        assert client._client._http.is_closed

    def test_thinking_logged_separately(self):
        """Thinking content is logged for traceability."""
        client = self._make_logged_client()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "<think>reasoning here</think>\nclean answer",
            "prompt_eval_count": 10,
            "eval_count": 20,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client._http, "post", return_value=mock_response):
            result = client.complete("test")

        assert result.text == "clean answer"
        assert result.thinking == "reasoning here"
        call = client.calls[0]
        assert call["response"] == "clean answer"
        assert call["thinking"] == "reasoning here"
        client.close()

    def test_no_thinking_uses_empty_string(self):
        """When no thinking block, 'thinking' key is present with empty string."""
        client = self._make_logged_client()
        with self._mock_http_response(client, "plain answer"):
            client.complete("test")

        call = client.calls[0]
        assert call["thinking"] == ""
        client.close()

    def test_config_passthrough(self):
        config = ModelConfig(
            model="qwen3:4b", base_url="http://localhost:11434",
            temperature=0.5, max_tokens=2048,
        )
        client = LoggedLLMClient(config)
        assert client.config.model == "qwen3:4b"
        assert client.config.temperature == 0.5
        assert client.config.max_tokens == 2048
        client.close()


class TestEnvironmentLLMClient:
    """Tests for EnvironmentLLMClient wrapper that auto-prepends environment brief."""

    def _make_inner(self):
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        return LoggedLLMClient(config)

    def _mock_http(self, inner, text="ok"):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": text,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_response.raise_for_status = MagicMock()
        return patch.object(inner._client._http, "post", return_value=mock_response)

    def test_prepends_brief_to_prompt(self):
        inner = self._make_inner()
        env_client = EnvironmentLLMClient(inner, "<environment>\nOS: Linux\n</environment>")

        with self._mock_http(inner):
            env_client.complete("do the thing", system="be helpful")

        # The logged call should contain the brief prepended to the prompt
        assert len(inner.calls) == 1
        logged_prompt = inner.calls[0]["prompt"]
        assert logged_prompt.startswith("<environment>")
        assert "do the thing" in logged_prompt

    def test_empty_brief_passes_through(self):
        inner = self._make_inner()
        env_client = EnvironmentLLMClient(inner, "")

        with self._mock_http(inner):
            env_client.complete("just the prompt")

        logged_prompt = inner.calls[0]["prompt"]
        assert logged_prompt == "just the prompt"

    def test_config_passthrough(self):
        inner = self._make_inner()
        env_client = EnvironmentLLMClient(inner, "brief")
        assert env_client.config.model == "qwen3:4b"

    def test_flush_delegates(self):
        inner = self._make_inner()
        env_client = EnvironmentLLMClient(inner, "brief")

        with self._mock_http(inner):
            env_client.complete("test")

        flushed = env_client.flush()
        assert len(flushed) == 1
        assert len(inner.calls) == 0  # flushed from inner

    def test_close_delegates(self):
        inner = self._make_inner()
        env_client = EnvironmentLLMClient(inner, "brief")
        env_client.close()
        assert inner._client._http.is_closed

    def test_context_manager(self):
        inner = self._make_inner()
        with EnvironmentLLMClient(inner, "brief") as env_client:
            assert env_client.config.model == "qwen3:4b"
        assert inner._client._http.is_closed
