"""Tests for cli.py."""

from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from clean_room_agent.cli import cli
from clean_room_agent.commands import resolve_budget, resolve_stages


class TestCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "cra" in result.output

    def test_init(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["init", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / ".clean_room" / "config.toml").exists()

    def test_init_creates_gitignore(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["init", str(tmp_path)])
        gitignore = tmp_path / ".gitignore"
        assert gitignore.exists()
        assert ".clean_room/" in gitignore.read_text()

    def test_init_existing_gitignore_already_has_marker(self, tmp_path):
        """init when .gitignore already contains '.clean_room/' should not duplicate it."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("node_modules/\n.clean_room/\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["init", str(tmp_path)])
        assert result.exit_code == 0
        content = gitignore.read_text()
        # Should not duplicate the marker
        assert content.count(".clean_room/") == 1

    def test_init_existing_gitignore_without_marker(self, tmp_path):
        """init when .gitignore exists but without '.clean_room/' appends it."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("node_modules/\n__pycache__/\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["init", str(tmp_path)])
        assert result.exit_code == 0
        content = gitignore.read_text()
        assert ".clean_room/" in content
        # Original content preserved
        assert "node_modules/" in content

    def test_init_already_exists(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["init", str(tmp_path)])
        result = runner.invoke(cli, ["init", str(tmp_path)])
        assert result.exit_code != 0

    def test_index_help(self):
        """Basic help test for the index command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["index", "--help"])
        assert result.exit_code == 0
        assert "--continue-on-error" in result.output
        assert "--verbose" in result.output

    def test_enrich_help(self):
        """Basic help test for the enrich command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["enrich", "--help"])
        assert result.exit_code == 0
        assert "--promote" in result.output

    def test_commands_registered(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "init" in result.output
        assert "index" in result.output
        assert "enrich" in result.output
        assert "retrieve" in result.output
        assert "plan" in result.output
        assert "solve" in result.output


class TestRetrieveCLI:
    def test_retrieve_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["retrieve", "--help"])
        assert result.exit_code == 0
        assert "--context-window" in result.output
        assert "--reserved-tokens" in result.output
        assert "--stages" in result.output
        assert "--plan" in result.output

    def test_retrieve_no_budget_no_config(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["retrieve", "fix bug", "--repo", str(tmp_path)])
        assert result.exit_code != 0
        assert "No config file found" in result.output

    def test_retrieve_no_stages_no_config(self, tmp_path):
        # Create config with budget but no stages
        clean_room = tmp_path / ".clean_room"
        clean_room.mkdir()
        (clean_room / "config.toml").write_text(
            '[models]\nprovider = "ollama"\ncoding = "m"\nreasoning = "m"\nbase_url = "http://x"\n'
            '[budget]\ncontext_window = 32768\nreserved_tokens = 4096\n'
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["retrieve", "fix bug", "--repo", str(tmp_path)])
        assert result.exit_code != 0
        assert "Stages not configured" in result.output

    def test_retrieve_budget_from_config(self, tmp_path):
        """Test that budget values are read from config.toml."""
        clean_room = tmp_path / ".clean_room"
        clean_room.mkdir()
        (clean_room / "config.toml").write_text(
            '[models]\nprovider = "ollama"\ncoding = "m"\nreasoning = "m"\nbase_url = "http://x"\n'
            '[budget]\ncontext_window = 32768\nreserved_tokens = 4096\n'
            '[stages]\ndefault = "scope,precision"\n'
        )
        runner = CliRunner()
        # This will fail at preflight (no curated DB) but proves budget/stages resolved
        result = runner.invoke(cli, ["retrieve", "fix bug", "--repo", str(tmp_path)])
        assert result.exit_code != 0
        # Should NOT say "Budget not configured" - it should fail later (preflight)
        assert "Budget not configured" not in result.output


class TestRetrieveModelsOnlyFallback:
    """A10: config with only [models].context_window and no [budget] section fails."""

    def test_models_only_context_window_fails(self, tmp_path):
        """A10: [models].context_window no longer used as fallback for retrieve budget."""
        clean_room = tmp_path / ".clean_room"
        clean_room.mkdir()
        (clean_room / "config.toml").write_text(
            '[models]\nprovider = "ollama"\ncoding = "m"\nreasoning = "m"\n'
            'base_url = "http://x"\ncontext_window = 32768\n'
            '[stages]\ndefault = "scope,precision"\n'
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["retrieve", "fix bug", "--repo", str(tmp_path)])
        assert result.exit_code != 0
        assert "Budget not configured" in result.output


class TestPlanCLI:
    def test_plan_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["plan", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output
        assert "--stages" in result.output

    def test_plan_no_config(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["plan", "Add feature", "--repo", str(tmp_path)])
        assert result.exit_code != 0
        assert "No config file found" in result.output

    def test_plan_no_budget(self, tmp_path):
        clean_room = tmp_path / ".clean_room"
        clean_room.mkdir()
        # Provide context_window in models but no reserved_tokens in budget
        (clean_room / "config.toml").write_text(
            '[models]\nprovider = "ollama"\ncoding = "m"\nreasoning = "m"\n'
            'base_url = "http://x"\ncontext_window = 32768\n'
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["plan", "Add feature", "--repo", str(tmp_path)])
        assert result.exit_code != 0
        assert "Budget not configured" in result.output


class TestSolveCLI:
    def test_solve_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["solve", "--help"])
        assert result.exit_code == 0
        assert "--plan" in result.output

    def test_solve_no_config(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["solve", "Fix bug", "--repo", str(tmp_path)])
        assert result.exit_code != 0
        assert "No config file found" in result.output

    def test_solve_no_testing_config(self, tmp_path):
        clean_room = tmp_path / ".clean_room"
        clean_room.mkdir()
        (clean_room / "config.toml").write_text(
            '[models]\nprovider = "ollama"\ncoding = "m"\nreasoning = "m"\nbase_url = "http://x"\n'
            'context_window = 32768\n'
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["solve", "Fix bug", "--repo", str(tmp_path)])
        assert result.exit_code != 0
        # Should fail on missing [testing] section (RuntimeError, not click output)
        assert result.exception is not None
        assert "testing" in str(result.exception).lower()


class TestResolveBudgetDirectly:
    """Test resolve_budget helper directly."""

    def testresolve_budget_directly(self):
        """Valid config returns (context_window, reserved_tokens) tuple."""
        config = {
            "models": {
                "provider": "ollama",
                "coding": "qwen2.5-coder:3b",
                "reasoning": "qwen3:4b",
                "base_url": "http://localhost:11434",
                "context_window": 32768,
            },
            "budget": {"reserved_tokens": 4096},
        }
        cw, rt = resolve_budget(config)
        assert cw == 32768
        assert rt == 4096

    def testresolve_budget_none_config_raises(self):
        """None config raises UsageError."""
        with pytest.raises(click.UsageError, match="Budget not configured"):
            resolve_budget(None)

    def testresolve_budget_no_reserved_tokens_raises(self):
        """Missing reserved_tokens raises UsageError."""
        config = {
            "models": {
                "provider": "ollama",
                "coding": "m",
                "reasoning": "m",
                "base_url": "http://x",
                "context_window": 32768,
            },
            "budget": {},
        }
        with pytest.raises(click.UsageError, match="reserved_tokens"):
            resolve_budget(config)

    def testresolve_budget_with_role(self):
        """Specifying role resolves the correct model's context_window."""
        config = {
            "models": {
                "provider": "ollama",
                "coding": "qwen2.5-coder:3b",
                "reasoning": "qwen3:4b",
                "base_url": "http://localhost:11434",
                "context_window": 32768,
            },
            "budget": {"reserved_tokens": 2048},
        }
        cw, rt = resolve_budget(config, role="coding")
        assert cw == 32768
        assert rt == 2048


class TestResolveStagesDirectly:
    """Test resolve_stages helper directly."""

    def testresolve_stages_directly(self):
        """Stages resolved from config default."""
        config = {"stages": {"default": "scope,precision"}}
        stages = resolve_stages(config, None)
        assert stages == ["scope", "precision"]

    def testresolve_stages_no_config_raises(self):
        """Missing config stages raises UsageError."""
        with pytest.raises(click.UsageError, match="Stages not configured"):
            resolve_stages({}, None)

    def testresolve_stages_none_config_raises(self):
        """None config raises UsageError."""
        with pytest.raises(click.UsageError, match="Stages not configured"):
            resolve_stages(None, None)

    def testresolve_stages_strips_whitespace(self):
        """Stage names are stripped of whitespace."""
        config = {"stages": {"default": " scope , precision , assembly "}}
        stages = resolve_stages(config, None)
        assert stages == ["scope", "precision", "assembly"]


class TestResolveStagesFromFlag:
    """Test resolve_stages with stages_flag parameter."""

    def testresolve_stages_from_flag(self):
        """CLI flag overrides config."""
        config = {"stages": {"default": "scope,precision"}}
        stages = resolve_stages(config, "custom_stage1,custom_stage2")
        assert stages == ["custom_stage1", "custom_stage2"]

    def testresolve_stages_flag_strips_whitespace(self):
        """Flag value whitespace is stripped."""
        stages = resolve_stages({}, " s1 , s2 ")
        assert stages == ["s1", "s2"]

    def testresolve_stages_flag_ignores_missing_config(self):
        """When flag is provided, missing config does not raise."""
        stages = resolve_stages(None, "scope,precision")
        assert stages == ["scope", "precision"]
