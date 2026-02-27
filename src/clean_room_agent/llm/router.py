"""Model routing: resolve (role, stage_name) -> ModelConfig."""

from clean_room_agent.llm.client import ModelConfig


class ModelRouter:
    """Resolves which model to use for a given role and optional stage."""

    _VALID_ROLES = ("coding", "reasoning", "classifier")

    # Roles that must have a model configured to use them.
    # "classifier" is Optional — when absent, stages that prefer it get "reasoning"
    # via explicit pipeline logic (not router fallback).
    _REQUIRED_ROLES = ("coding", "reasoning")

    def __init__(self, models_config: dict):
        self._coding = models_config.get("coding")
        self._reasoning = models_config.get("reasoning")
        self._classifier = models_config.get("classifier")
        self._base_url = models_config.get("base_url")

        overrides = models_config["overrides"]
        if not isinstance(overrides, dict):
            raise RuntimeError(
                f"'overrides' in [models] config must be a dict, got {type(overrides).__name__}"
            )
        self._overrides = overrides

        if "context_window" not in models_config:
            raise RuntimeError("Missing 'context_window' in [models] config")

        # Determine which roles are active (have a model tag configured).
        # Required roles can still be absent — they'll fail-fast at resolve() time.
        self._active_roles: list[str] = []
        if self._coding:
            self._active_roles.append("coding")
        if self._reasoning:
            self._active_roles.append("reasoning")
        if self._classifier:
            self._active_roles.append("classifier")

        # context_window per role: int (all active roles) or dict with per-role keys
        cw = models_config["context_window"]
        if isinstance(cw, int):
            self._context_window = {r: cw for r in self._active_roles}
        elif isinstance(cw, dict):
            self._context_window = {}
            for role in self._active_roles:
                if role not in cw:
                    raise RuntimeError(
                        f"context_window dict missing '{role}' key. "
                        f"Provide all active role keys, or use a single int."
                    )
                self._context_window[role] = cw[role]
        else:
            raise RuntimeError(
                f"'context_window' must be an int or dict, got {type(cw).__name__}"
            )

        if "provider" not in models_config:
            raise RuntimeError("Missing 'provider' in [models] config")
        self._provider = models_config["provider"]

        # Temperature per role: 0.0 (deterministic) is the safe default
        temps = models_config["temperature"]
        self._temperature = {
            "coding": temps["coding"],
            "reasoning": temps["reasoning"],
            "classifier": temps["classifier"],
        }

        # max_tokens per role: optional, derived from context_window // 8 when absent
        mt = models_config.get("max_tokens")
        if mt is None:
            self._max_tokens = {r: self._context_window[r] // 8 for r in self._active_roles}
        elif isinstance(mt, int):
            self._max_tokens = {r: mt for r in self._active_roles}
        elif isinstance(mt, dict):
            self._max_tokens = {}
            for role in self._active_roles:
                if role not in mt:
                    raise RuntimeError(
                        f"max_tokens dict missing '{role}' key. "
                        f"Provide all active role keys, or use a single int."
                    )
                self._max_tokens[role] = mt[role]
        else:
            raise RuntimeError(
                f"'max_tokens' must be an int or dict, got {type(mt).__name__}"
            )

        if not self._base_url:
            raise RuntimeError("Missing 'base_url' in [models] config")

    def has_role(self, role: str) -> bool:
        """Check if a role is configured (has a model assigned)."""
        if role not in self._VALID_ROLES:
            raise ValueError(
                f"Unknown role: {role!r}. Must be one of {self._VALID_ROLES}."
            )
        return role in self._active_roles

    def resolve(self, role: str, stage_name: str | None = None) -> ModelConfig:
        """Resolve to a ModelConfig.

        Resolution order:
        1. [models.overrides] has a key matching stage_name -> use that model tag
           Override can be a string (model tag) or dict with 'model' and optional 'context_window'.
        2. No stage override -> use [models].<role> (coding or reasoning or classifier)
        3. [models] section missing the requested role -> hard error (no fallbacks)

        context_window resolution: override-specific > role-specific > (fail-fast if missing)
        """
        if role not in self._VALID_ROLES:
            raise ValueError(
                f"Unknown role: {role!r}. Must be one of {self._VALID_ROLES}."
            )

        # Fail-fast: requested role must be configured
        if role not in self._active_roles:
            raise RuntimeError(
                f"Role '{role}' requested but not configured in [models]. "
                f"Add '{role} = \"<model_tag>\"' to [models] in config.toml."
            )

        # Check stage override first
        if stage_name and stage_name in self._overrides:
            override = self._overrides[stage_name]
            if isinstance(override, str):
                model_tag = override
                cw = self._context_window[role]
                mt = self._max_tokens[role]
            elif isinstance(override, dict):
                if "model" not in override:
                    raise RuntimeError(
                        f"Override for stage '{stage_name}' is a dict but missing 'model' key."
                    )
                model_tag = override["model"]
                cw = override.get("context_window", self._context_window[role])
                mt = override.get("max_tokens", self._max_tokens[role])
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
                max_tokens=mt,
                context_window=cw,
            )

        # Role-based default
        role_to_tag = {
            "coding": self._coding,
            "reasoning": self._reasoning,
            "classifier": self._classifier,
        }
        model_tag = role_to_tag[role]
        return ModelConfig(
            model=model_tag,
            base_url=self._base_url,
            provider=self._provider,
            temperature=self._temperature[role],
            max_tokens=self._max_tokens[role],
            context_window=self._context_window[role],
        )
