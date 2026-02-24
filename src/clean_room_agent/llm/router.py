"""Model routing: resolve (role, stage_name) -> ModelConfig."""

from clean_room_agent.llm.client import ModelConfig


class ModelRouter:
    """Resolves which model to use for a given role and optional stage."""

    _VALID_ROLES = ("coding", "reasoning")

    def __init__(self, models_config: dict):
        self._coding = models_config.get("coding")
        self._reasoning = models_config.get("reasoning")
        self._base_url = models_config.get("base_url")

        overrides = models_config.get("overrides", {})
        if not isinstance(overrides, dict):
            raise RuntimeError(
                f"'overrides' in [models] config must be a dict, got {type(overrides).__name__}"
            )
        self._overrides = overrides

        if "context_window" not in models_config:
            raise RuntimeError("Missing 'context_window' in [models] config")
        self._context_window = models_config["context_window"]

        if "provider" not in models_config:
            raise RuntimeError("Missing 'provider' in [models] config")
        self._provider = models_config["provider"]

        # Temperature per role: 0.0 (deterministic) is the safe default
        temps = models_config.get("temperature", {})
        self._temperature = {
            "coding": temps.get("coding", 0.0),
            "reasoning": temps.get("reasoning", 0.0),
        }

        # max_tokens per role: must be int or dict with both role keys
        mt = models_config.get("max_tokens", {"coding": 4096, "reasoning": 4096})
        if isinstance(mt, int):
            self._max_tokens = {"coding": mt, "reasoning": mt}
        elif isinstance(mt, dict):
            for role in self._VALID_ROLES:
                if role not in mt:
                    raise RuntimeError(
                        f"max_tokens dict missing '{role}' key. "
                        f"Provide both 'coding' and 'reasoning', or use a single int."
                    )
            self._max_tokens = {"coding": mt["coding"], "reasoning": mt["reasoning"]}
        else:
            raise RuntimeError(
                f"'max_tokens' must be an int or dict, got {type(mt).__name__}"
            )

        if not self._base_url:
            raise RuntimeError("Missing 'base_url' in [models] config")

    def resolve(self, role: str, stage_name: str | None = None) -> ModelConfig:
        """Resolve to a ModelConfig.

        Resolution order:
        1. [models.overrides] has a key matching stage_name -> use that model tag
        2. No stage override -> use [models].<role> (coding or reasoning)
        3. [models] section missing the role -> hard error
        """
        if role not in self._VALID_ROLES:
            raise ValueError(f"Unknown role: {role!r}. Must be 'coding' or 'reasoning'.")

        # Check stage override first
        if stage_name and stage_name in self._overrides:
            model_tag = self._overrides[stage_name]
            return ModelConfig(
                model=model_tag,
                base_url=self._base_url,
                provider=self._provider,
                temperature=self._temperature[role],
                max_tokens=self._max_tokens[role],
                context_window=self._context_window,
            )

        # Role-based default
        model_tag = self._coding if role == "coding" else self._reasoning
        if not model_tag:
            raise RuntimeError(f"Missing '{role}' model in [models] config")
        return ModelConfig(
            model=model_tag,
            base_url=self._base_url,
            provider=self._provider,
            temperature=self._temperature[role],
            max_tokens=self._max_tokens[role],
            context_window=self._context_window,
        )
