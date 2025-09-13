from typer.testing import CliRunner
import typer

# Import the CLI function without executing it
from denoise import cli as cli_command


def test_cli_help_renders():
    app = typer.Typer()
    app.command()(cli_command)
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])  # help should not run the pipeline
    assert result.exit_code == 0
    assert "Usage" in result.stdout or "Usage" in result.output
