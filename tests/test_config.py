"""Tests for config.py."""

import tomllib

import pytest

from clean_room_agent.config import create_default_config, load_config, require_models_config


class TestLoadConfig:
    def test_returns_none_when_no_config(self, tmp_path):
        assert load_config(tmp_path) is None

    def test_malformed_toml_raises(self, tmp_path):
        config_dir = tmp_path / ".clean_room"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text("invalid [[ toml ===")
        with pytest.raises(tomllib.TOMLDecodeError):
            load_config(tmp_path)

    def test_loads_valid_toml(self, tmp_path):
        config_dir = tmp_path / ".clean_room"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            '[models]\nprovider = "ollama"\ncoding = "qwen2.5-coder:3b"\n'
        )
        config = load_config(tmp_path)
        assert config is not None
        assert config["models"]["provider"] == "ollama"
        assert config["models"]["coding"] == "qwen2.5-coder:3b"

    def test_loads_config_with_all_sections(self, tmp_path):
        config_dir = tmp_path / ".clean_room"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            '[models]\nprovider = "ollama"\ncoding = "x"\nreasoning = "y"\n'
            'base_url = "http://localhost:11434"\n'
            "\n[models.overrides]\nscope = \"custom-model\"\n"
        )
        config = load_config(tmp_path)
        assert config["models"]["overrides"]["scope"] == "custom-model"


class TestRequireModelsConfig:
    def test_raises_on_none_config(self):
        with pytest.raises(RuntimeError, match="No config file found"):
            require_models_config(None)

    def test_raises_on_missing_models_section(self):
        with pytest.raises(RuntimeError, match="Missing \\[models\\]"):
            require_models_config({"other": "stuff"})

    def test_returns_models_dict(self):
        config = {"models": {"provider": "ollama", "coding": "x"}}
        result = require_models_config(config)
        assert result["provider"] == "ollama"
        assert result["coding"] == "x"


class TestCreateDefaultConfig:
    def test_creates_file(self, tmp_path):
        path = create_default_config(tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "[models]" in content
        assert "ollama" in content

    def test_creates_clean_room_directory(self, tmp_path):
        create_default_config(tmp_path)
        assert (tmp_path / ".clean_room").is_dir()

    def test_raises_if_exists(self, tmp_path):
        create_default_config(tmp_path)
        with pytest.raises(FileExistsError, match="already exists"):
            create_default_config(tmp_path)

    def test_roundtrip_with_load(self, tmp_path):
        create_default_config(tmp_path)
        config = load_config(tmp_path)
        assert config is not None
        models = require_models_config(config)
        assert models["provider"] == "ollama"
        assert models["coding"] == "qwen2.5-coder:3b-instruct"
        assert models["reasoning"] == "qwen3:4b-instruct-2507"
        assert models["base_url"] == "http://localhost:11434"

    def test_default_config_has_budget_section(self, tmp_path):
        create_default_config(tmp_path)
        config = load_config(tmp_path)
        assert "budget" in config
        # context_window now derives from [models].context_window (single source of truth)
        assert "context_window" not in config["budget"]
        assert config["budget"]["reserved_tokens"] == 4096

    def test_default_config_has_stages_section(self, tmp_path):
        create_default_config(tmp_path)
        config = load_config(tmp_path)
        assert "stages" in config
        assert config["stages"]["default"] == "scope,precision"
