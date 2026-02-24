"""Tests for cli.py."""

from unittest.mock import patch

from click.testing import CliRunner

from clean_room_agent.cli import cli


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

    def test_init_already_exists(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["init", str(tmp_path)])
        result = runner.invoke(cli, ["init", str(tmp_path)])
        assert result.exit_code != 0

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
        assert "Budget not configured" in result.output

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
