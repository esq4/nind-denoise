def test_typer_help_displays_usage():
    import typer
    from typer.testing import CliRunner
    import denoise

    app = typer.Typer()
    app.command()(denoise.cli)

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])    
    assert result.exit_code == 0
    assert "Usage" in result.stdout or "Usage:" in result.stdout
