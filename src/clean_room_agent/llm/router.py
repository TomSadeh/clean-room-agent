"""Model routing: resolve (role, stage_name) -> ModelConfig."""

from clean_room_agent.llm.client import ModelConfig


class ModelRouter:
    """Resolves which model to use for a given role and optional stage."""

    def __init__(self, models_config: dict):
        self._coding = models_config.get("coding")
        self._reasoning = models_config.get("reasoning")
        self._base_url = models_config.get("base_url")
        self._overrides = models_config.get("overrides", {})

        if "provider" not in models_config:
            raise RuntimeError("Missing 'provider' in [models] config")
        self._provider = models_config["provider"]

        # Temperature per role: read from config, default to 0.0 (deterministic)
        temps = models_config.get("temperature", {})
        self._temperature = {
            "coding": temps.get("coding", 0.0),
            "reasoning": temps.get("reasoning", 0.0),
        }

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
            )
        if role == "reasoning":
            if not self._reasoning:
                raise RuntimeError("Missing 'reasoning' model in [models] config")
            return ModelConfig(
                model=self._reasoning,
                base_url=self._base_url,
                provider=self._provider,
                temperature=self._temperature.get(role, 0.0),
            )

        raise ValueError(f"Unknown role: {role!r}. Must be 'coding' or 'reasoning'.")
