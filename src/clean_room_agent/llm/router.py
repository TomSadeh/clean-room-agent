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

        # context_window per role: int (both roles) or dict with both role keys
        cw = models_config["context_window"]
        if isinstance(cw, int):
            self._context_window = {"coding": cw, "reasoning": cw}
        elif isinstance(cw, dict):
            for role in self._VALID_ROLES:
                if role not in cw:
                    raise RuntimeError(
                        f"context_window dict missing '{role}' key. "
                        f"Provide both 'coding' and 'reasoning', or use a single int."
                    )
            self._context_window = {"coding": cw["coding"], "reasoning": cw["reasoning"]}
        else:
            raise RuntimeError(
                f"'context_window' must be an int or dict, got {type(cw).__name__}"
            )

        if "provider" not in models_config:
            raise RuntimeError("Missing 'provider' in [models] config")
        self._provider = models_config["provider"]

        # Temperature per role: 0.0 (deterministic) is the safe default
        temps = models_config.get("temperature", {})
        self._temperature = {
            "coding": temps.get("coding", 0.0),
            "reasoning": temps.get("reasoning", 0.0),
        }

        # max_tokens per role: optional, derived from context_window // 8 when absent
        mt = models_config.get("max_tokens")
        if mt is None:
            mt = {r: self._context_window[r] // 8 for r in self._VALID_ROLES}
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
           Override can be a string (model tag) or dict with 'model' and optional 'context_window'.
        2. No stage override -> use [models].<role> (coding or reasoning)
        3. [models] section missing the role -> hard error

        context_window resolution: override-specific > role-specific > (fail-fast if missing)
        """
        if role not in self._VALID_ROLES:
            raise ValueError(f"Unknown role: {role!r}. Must be 'coding' or 'reasoning'.")

        # Check stage override first
        if stage_name and stage_name in self._overrides:
            override = self._overrides[stage_name]
            if isinstance(override, str):
                model_tag = override
                cw = self._context_window[role]
            elif isinstance(override, dict):
                if "model" not in override:
                    raise RuntimeError(
                        f"Override for stage '{stage_name}' is a dict but missing 'model' key."
                    )
                model_tag = override["model"]
                cw = override.get("context_window", self._context_window[role])
            else:
                raise RuntimeError(
                    f"Override for stage '{stage_name}' must be a string or dict, "
                    f"got {type(override).__name__}"
                )
            return ModelConfig(
                model=model_tag,
                base_url=self._base_url,
                provider=self._provider,
                temperature=self._temperature[role],
                max_tokens=self._max_tokens[role],
                context_window=cw,
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
            context_window=self._context_window[role],
        )
