"""Tests for llm/client.py."""

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


class TestLLMClient:
    def test_instantiation(self):
        config = ModelConfig(model="qwen3:4b", base_url="http://localhost:11434")
        client = LLMClient(config)
        assert client.config.model == "qwen3:4b"

    def test_non_ollama_provider_fails_fast(self):
        config = ModelConfig(
            model="qwen3:4b",
            base_url="http://localhost:11434",
            provider="openai_compat",
        )
        client = LLMClient(config)
        import pytest
        with pytest.raises(RuntimeError, match="not implemented"):
            client.complete("hello")
