"""Tests for llm/router.py."""

import pytest

from clean_room_agent.llm.router import ModelRouter


class TestModelRouter:
    def setup_method(self):
        self.config = {
            "coding": "qwen2.5-coder:3b",
            "reasoning": "qwen3:4b",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "overrides": {
                "scope": "qwen3:4b-scope-v1",
            },
        }

    def test_resolve_coding(self):
        router = ModelRouter(self.config)
        mc = router.resolve("coding")
        assert mc.model == "qwen2.5-coder:3b"
        assert mc.base_url == "http://localhost:11434"
        assert mc.provider == "ollama"

    def test_resolve_reasoning(self):
        router = ModelRouter(self.config)
        mc = router.resolve("reasoning")
        assert mc.model == "qwen3:4b"

    def test_stage_override(self):
        router = ModelRouter(self.config)
        mc = router.resolve("reasoning", stage_name="scope")
        assert mc.model == "qwen3:4b-scope-v1"

    def test_no_override_falls_through(self):
        router = ModelRouter(self.config)
        mc = router.resolve("reasoning", stage_name="precision")
        assert mc.model == "qwen3:4b"

    def test_missing_role(self):
        router = ModelRouter(self.config)
        with pytest.raises(ValueError, match="Unknown role"):
            router.resolve("unknown")

    def test_missing_base_url(self):
        with pytest.raises(RuntimeError, match="base_url"):
            ModelRouter({"coding": "x", "reasoning": "y", "provider": "ollama"})

    def test_missing_coding_model(self):
        router = ModelRouter({"reasoning": "y", "provider": "ollama", "base_url": "http://localhost:11434"})
        with pytest.raises(RuntimeError, match="coding"):
            router.resolve("coding")

    def test_missing_provider(self):
        with pytest.raises(RuntimeError, match="provider"):
            ModelRouter({"coding": "x", "reasoning": "y", "base_url": "http://localhost:11434"})

    def test_provider_passthrough(self):
        router = ModelRouter(
            {
                "coding": "qwen2.5-coder:3b",
                "reasoning": "qwen3:4b",
                "base_url": "http://localhost:11434",
                "provider": "openai_compat",
            }
        )
        mc = router.resolve("reasoning")
        assert mc.provider == "openai_compat"

    def test_custom_temperature(self):
        config = {
            **self.config,
            "temperature": {"coding": 0.2, "reasoning": 0.7},
        }
        router = ModelRouter(config)
        mc_coding = router.resolve("coding")
        mc_reasoning = router.resolve("reasoning")
        assert mc_coding.temperature == 0.2
        assert mc_reasoning.temperature == 0.7

    def test_default_temperature_is_zero(self):
        router = ModelRouter(self.config)
        mc = router.resolve("coding")
        assert mc.temperature == 0.0

    def test_override_inherits_role_temperature(self):
        config = {
            **self.config,
            "temperature": {"reasoning": 0.5},
        }
        router = ModelRouter(config)
        mc = router.resolve("reasoning", stage_name="scope")
        assert mc.model == "qwen3:4b-scope-v1"
        assert mc.temperature == 0.5
