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
            "context_window": 32768,
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
            ModelRouter({"coding": "x", "reasoning": "y", "provider": "ollama", "context_window": 32768})

    def test_missing_coding_model(self):
        router = ModelRouter({"reasoning": "y", "provider": "ollama", "base_url": "http://localhost:11434", "context_window": 32768})
        with pytest.raises(RuntimeError, match="coding"):
            router.resolve("coding")

    def test_missing_context_window(self):
        with pytest.raises(RuntimeError, match="context_window"):
            ModelRouter({"coding": "x", "reasoning": "y", "provider": "ollama", "base_url": "http://localhost:11434"})

    def test_missing_provider(self):
        with pytest.raises(RuntimeError, match="provider"):
            ModelRouter({"coding": "x", "reasoning": "y", "base_url": "http://localhost:11434", "context_window": 32768})

    def test_provider_passthrough(self):
        router = ModelRouter(
            {
                "coding": "qwen2.5-coder:3b",
                "reasoning": "qwen3:4b",
                "base_url": "http://localhost:11434",
                "provider": "openai_compat",
                "context_window": 32768,
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
            "temperature": {"coding": 0.0, "reasoning": 0.5},
        }
        router = ModelRouter(config)
        mc = router.resolve("reasoning", stage_name="scope")
        assert mc.model == "qwen3:4b-scope-v1"
        assert mc.temperature == 0.5

    def test_max_tokens_int(self):
        config = {**self.config, "max_tokens": 2048}
        router = ModelRouter(config)
        mc = router.resolve("coding")
        assert mc.max_tokens == 2048
        mc2 = router.resolve("reasoning")
        assert mc2.max_tokens == 2048

    def test_max_tokens_dict(self):
        config = {**self.config, "max_tokens": {"coding": 1024, "reasoning": 2048}}
        router = ModelRouter(config)
        assert router.resolve("coding").max_tokens == 1024
        assert router.resolve("reasoning").max_tokens == 2048

    def test_max_tokens_dict_missing_key_raises(self):
        """T7: incomplete max_tokens dict is a config error, not silently defaulted."""
        config = {**self.config, "max_tokens": {"coding": 1024}}
        with pytest.raises(RuntimeError, match="max_tokens dict missing 'reasoning'"):
            ModelRouter(config)

    def test_max_tokens_invalid_type_raises(self):
        """T7: unrecognized max_tokens type must raise, not silently default."""
        config = {**self.config, "max_tokens": "4096"}
        with pytest.raises(RuntimeError, match="must be an int or dict"):
            ModelRouter(config)

    def test_overrides_invalid_type_raises(self):
        """T7: overrides must be a dict."""
        config = {**self.config, "overrides": ["scope", "precision"]}
        with pytest.raises(RuntimeError, match="must be a dict"):
            ModelRouter(config)

    def test_invalid_role_in_stage_override(self):
        """T7: invalid role caught even when stage override matches."""
        router = ModelRouter(self.config)
        with pytest.raises(ValueError, match="Unknown role"):
            router.resolve("invalid_role", stage_name="scope")
