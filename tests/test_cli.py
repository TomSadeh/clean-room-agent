"""Tests for cli.py."""

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
