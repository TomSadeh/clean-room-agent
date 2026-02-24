"""Model routing: resolve (role, stage_name) -> ModelConfig."""

from clean_room_agent.llm.client import ModelConfig


class ModelRouter:
    """Resolves which model to use for a given role and optional stage."""

    def __init__(self, models_config: dict):
        self._coding = models_config.get("coding")
        self._reasoning = models_config.get("reasoning")
        self._base_url = models_config.get("base_url")
        self._overrides = models_config.get("overrides", {})
        if "context_window" not in models_config:
            raise RuntimeError("Missing 'context_window' in [models] config")
        self._context_window = models_config["context_window"]

        if "provider" not in models_config:
            raise RuntimeError("Missing 'provider' in [models] config")
        self._provider = models_config["provider"]

        # Temperature per role: read from config, default to 0.0 (deterministic)
        temps = models_config.get("temperature", {})
        self._temperature = {
            "coding": temps.get("coding", 0.0),
            "reasoning": temps.get("reasoning", 0.0),
        }

        # max_tokens per role: configurable
        mt = models_config.get("max_tokens", {})
        if isinstance(mt, int):
            # Flat value applies to both roles
            self._max_tokens = {"coding": mt, "reasoning": mt}
        elif isinstance(mt, dict):
            self._max_tokens = {
                "coding": mt.get("coding", 4096),
                "reasoning": mt.get("reasoning", 4096),
            }
        else:
            self._max_tokens = {"coding": 4096, "reasoning": 4096}

        if not self._base_url:
            raise RuntimeError("Missing 'base_url' in [models] config")

    def resolve(self, role: str, stage_name: str | None = None) -> ModelConfig:
        """Resolve to a ModelConfig.

        Resolution order:
        1. [models.overrides] has a key matching stage_name -> use that model tag
        2. No stage override -> use [models].<role> (coding or reasoning)
        3. [models] section missing the role -> hard error
        """
        # Check stage override first
        if stage_name and stage_name in self._overrides:
            model_tag = self._overrides[stage_name]
            return ModelConfig(
                model=model_tag,
                base_url=self._base_url,
                provider=self._provider,
                temperature=self._temperature.get(role, 0.0),
                max_tokens=self._max_tokens.get(role, 4096),
                context_window=self._context_window,
            )

        # Role-based default
        if role == "coding":
            if not self._coding:
                raise RuntimeError("Missing 'coding' model in [models] config")
            return ModelConfig(
                model=self._coding,
                base_url=self._base_url,
                provider=self._provider,
                temperature=self._temperature.get(role, 0.0),
                max_tokens=self._max_tokens.get(role, 4096),
                context_window=self._context_window,
            )
        if role == "reasoning":
            if not self._reasoning:
                raise RuntimeError("Missing 'reasoning' model in [models] config")
            return ModelConfig(
                model=self._reasoning,
                base_url=self._base_url,
                provider=self._provider,
                temperature=self._temperature.get(role, 0.0),
                max_tokens=self._max_tokens.get(role, 4096),
                context_window=self._context_window,
            )

        raise ValueError(f"Unknown role: {role!r}. Must be 'coding' or 'reasoning'.")
