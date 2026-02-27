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
            "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0},
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
            ModelRouter({"coding": "x", "reasoning": "y", "provider": "ollama", "context_window": 32768, "overrides": {}, "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0}})

    def test_missing_coding_model(self):
        router = ModelRouter({"reasoning": "y", "provider": "ollama", "base_url": "http://localhost:11434", "context_window": 32768, "overrides": {}, "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0}})
        with pytest.raises(RuntimeError, match="coding"):
            router.resolve("coding")

    def test_resolve_reasoning_not_configured(self):
        """resolve('reasoning') when reasoning key is missing from config should raise RuntimeError."""
        router = ModelRouter({"coding": "x", "provider": "ollama", "base_url": "http://localhost:11434", "context_window": 32768, "overrides": {}, "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0}})
        with pytest.raises(RuntimeError, match="reasoning"):
            router.resolve("reasoning")

    def test_max_tokens_derived_from_context_window(self):
        """H3: max_tokens defaults to context_window // 8 when omitted."""
        router = ModelRouter({
            "coding": "x", "reasoning": "y", "provider": "ollama",
            "base_url": "http://localhost:11434", "context_window": 32768,
            "overrides": {}, "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0},
        })
        assert router.resolve("coding").max_tokens == 4096
        assert router.resolve("reasoning").max_tokens == 4096

    def test_max_tokens_derived_per_role(self):
        """H3: derivation respects per-role context_window."""
        router = ModelRouter({
            "coding": "x", "reasoning": "y", "provider": "ollama",
            "base_url": "http://localhost:11434",
            "context_window": {"coding": 16384, "reasoning": 32768},
            "overrides": {}, "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0},
        })
        assert router.resolve("coding").max_tokens == 2048
        assert router.resolve("reasoning").max_tokens == 4096

    def test_missing_context_window(self):
        with pytest.raises(RuntimeError, match="context_window"):
            ModelRouter({"coding": "x", "reasoning": "y", "provider": "ollama", "base_url": "http://localhost:11434", "overrides": {}, "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0}})

    def test_missing_provider(self):
        with pytest.raises(RuntimeError, match="provider"):
            ModelRouter({"coding": "x", "reasoning": "y", "base_url": "http://localhost:11434", "context_window": 32768, "overrides": {}, "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0}})

    def test_provider_passthrough(self):
        router = ModelRouter(
            {
                "coding": "qwen2.5-coder:3b",
                "reasoning": "qwen3:4b",
                "base_url": "http://localhost:11434",
                "provider": "openai_compat",
                "context_window": 32768,
                "overrides": {},
                "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0},
            }
        )
        mc = router.resolve("reasoning")
        assert mc.provider == "openai_compat"

    def test_custom_temperature(self):
        config = {
            **self.config,
            "temperature": {"coding": 0.2, "reasoning": 0.7, "classifier": 0.0},
        }
        router = ModelRouter(config)
        mc_coding = router.resolve("coding")
        mc_reasoning = router.resolve("reasoning")
        assert mc_coding.temperature == 0.2
        assert mc_reasoning.temperature == 0.7

    def test_explicit_temperature_zero(self):
        router = ModelRouter(self.config)
        mc = router.resolve("coding")
        assert mc.temperature == 0.0

    def test_missing_temperature_defaults_to_zero(self):
        """C2: config missing 'temperature' key defaults all roles to 0.0."""
        config = {k: v for k, v in self.config.items() if k != "temperature"}
        router = ModelRouter(config)
        assert router.resolve("coding").temperature == 0.0
        assert router.resolve("reasoning").temperature == 0.0

    def test_temperature_missing_subkey_defaults(self):
        """C2: temperature dict missing 'coding' key defaults to 0.0."""
        config = {**self.config, "temperature": {"reasoning": 0.5, "classifier": 0.0}}
        router = ModelRouter(config)
        assert router.resolve("coding").temperature == 0.0
        assert router.resolve("reasoning").temperature == 0.5

    def test_temperature_missing_classifier_defaults(self):
        """C2: temperature dict missing 'classifier' key defaults to 0.0."""
        config = {**self.config, "temperature": {"coding": 0.2, "reasoning": 0.0}}
        router = ModelRouter(config)
        assert router.resolve("coding").temperature == 0.2

    def test_override_inherits_role_temperature(self):
        config = {
            **self.config,
            "temperature": {"coding": 0.0, "reasoning": 0.5, "classifier": 0.0},
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

    def test_missing_overrides_defaults_to_empty(self):
        """C1: config without 'overrides' defaults to empty dict."""
        config = {k: v for k, v in self.config.items() if k != "overrides"}
        router = ModelRouter(config)
        # No override → role default
        mc = router.resolve("reasoning", stage_name="scope")
        assert mc.model == "qwen3:4b"

    def test_invalid_role_in_stage_override(self):
        """T7: invalid role caught even when stage override matches."""
        router = ModelRouter(self.config)
        with pytest.raises(ValueError, match="Unknown role"):
            router.resolve("invalid_role", stage_name="scope")


class TestContextWindowPerRole:
    """T21: context_window must be per-role, not global-only."""

    def setup_method(self):
        self.base = {
            "coding": "qwen2.5-coder:3b",
            "reasoning": "qwen3:4b",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "overrides": {},
            "temperature": {"coding": 0.0, "reasoning": 0.0, "classifier": 0.0},
        }

    def test_int_applies_to_both_roles(self):
        config = {**self.base, "context_window": 32768}
        router = ModelRouter(config)
        assert router.resolve("coding").context_window == 32768
        assert router.resolve("reasoning").context_window == 32768

    def test_dict_per_role(self):
        config = {**self.base, "context_window": {"coding": 16384, "reasoning": 32768}}
        router = ModelRouter(config)
        assert router.resolve("coding").context_window == 16384
        assert router.resolve("reasoning").context_window == 32768

    def test_dict_missing_role_raises(self):
        config = {**self.base, "context_window": {"coding": 16384}}
        with pytest.raises(RuntimeError, match="context_window dict missing 'reasoning'"):
            ModelRouter(config)

    def test_invalid_type_raises(self):
        config = {**self.base, "context_window": "32768"}
        with pytest.raises(RuntimeError, match="must be an int or dict"):
            ModelRouter(config)

    def test_override_inherits_role_context_window(self):
        """Override without context_window uses the role's value."""
        config = {
            **self.base,
            "context_window": {"coding": 16384, "reasoning": 32768},
            "overrides": {"scope": "qwen3:4b-scope-v1"},
        }
        router = ModelRouter(config)
        mc = router.resolve("reasoning", stage_name="scope")
        assert mc.model == "qwen3:4b-scope-v1"
        assert mc.context_window == 32768

    def test_override_dict_with_context_window(self):
        """Override dict can specify its own context_window."""
        config = {
            **self.base,
            "context_window": 32768,
            "overrides": {
                "scope": {"model": "qwen3:4b-scope-v1", "context_window": 65536},
            },
        }
        router = ModelRouter(config)
        mc = router.resolve("reasoning", stage_name="scope")
        assert mc.model == "qwen3:4b-scope-v1"
        assert mc.context_window == 65536

    def test_override_dict_without_context_window_inherits(self):
        """Override dict without context_window falls back to role."""
        config = {
            **self.base,
            "context_window": {"coding": 16384, "reasoning": 32768},
            "overrides": {
                "scope": {"model": "qwen3:4b-scope-v1"},
            },
        }
        router = ModelRouter(config)
        mc = router.resolve("reasoning", stage_name="scope")
        assert mc.context_window == 32768

    def test_override_dict_with_max_tokens(self):
        """Override dict can specify max_tokens (e.g. 16 for binary classifiers)."""
        config = {
            **self.base,
            "context_window": 32768,
            "overrides": {
                "scope": {"model": "qwen3:0.6b", "context_window": 8192, "max_tokens": 16},
            },
        }
        router = ModelRouter(config)
        mc = router.resolve("reasoning", stage_name="scope")
        assert mc.model == "qwen3:0.6b"
        assert mc.context_window == 8192
        assert mc.max_tokens == 16

    def test_override_dict_without_max_tokens_inherits(self):
        """Override dict without max_tokens falls back to role's max_tokens."""
        config = {
            **self.base,
            "context_window": 32768,
            "overrides": {
                "scope": {"model": "qwen3:0.6b"},
            },
        }
        router = ModelRouter(config)
        mc = router.resolve("reasoning", stage_name="scope")
        assert mc.max_tokens == 32768 // 8  # derived from role's context_window

    def test_override_string_inherits_max_tokens(self):
        """String override inherits max_tokens from role."""
        config = {
            **self.base,
            "context_window": 32768,
            "overrides": {"scope": "qwen3:4b-scope-v1"},
        }
        router = ModelRouter(config)
        mc = router.resolve("reasoning", stage_name="scope")
        assert mc.max_tokens == 32768 // 8

    def test_override_dict_missing_model_raises(self):
        config = {
            **self.base,
            "context_window": 32768,
            "overrides": {"scope": {"context_window": 65536}},
        }
        router = ModelRouter(config)
        with pytest.raises(RuntimeError, match="missing 'model' key"):
            router.resolve("reasoning", stage_name="scope")

    def test_override_invalid_type_raises(self):
        config = {
            **self.base,
            "context_window": 32768,
            "overrides": {"scope": 42},
        }
        router = ModelRouter(config)
        with pytest.raises(RuntimeError, match="must be a string or dict"):
            router.resolve("reasoning", stage_name="scope")
