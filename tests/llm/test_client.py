"""Tests for llm/client.py."""

from unittest.mock import MagicMock, patch

import pytest

from clean_room_agent.llm.client import LLMClient, LLMResponse, ModelConfig


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
        resp = LLMResponse(text="hello", prompt_tokens=10, completion_tokens=5, latency_ms=100)
        assert resp.text == "hello"
        assert resp.prompt_tokens == 10

    def test_tokens_can_be_none(self):
        resp = LLMResponse(text="hello", prompt_tokens=None, completion_tokens=None, latency_ms=50)
        assert resp.prompt_tokens is None
        assert resp.completion_tokens is None


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
            temperature=0.3, max_tokens=1024,
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
        assert payload["options"]["num_predict"] == 1024

        assert result.text == "Hello world"
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 8
        assert result.latency_ms >= 0
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
