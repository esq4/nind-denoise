from src.nind_denoise.exceptions import ExternalToolNotFound


def test_config_initialization(tmp_path):
    # Arrange
    config_content = """
models:
  model1:
    path: "path/to/model1"
    default: true
tools:
  posix:
    tool1:
      path: "/usr/bin/tool1"
      args: ["-arg1"]
operations:
  operations:
    first_stage: ["op1", "op2"]
    second_stage: ["op3", "op4"]
nightmode_ops: ["op5"]
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    # Act
    config = Config(path=config_file, verbose=True)

    # Assert
    assert config.verbose is True


def test_config_models_property(tmp_path):
    # Arrange
    config_content = """
models:
  model1:
    path: "path/to/model1"
    default: true
tools:
  posix:
    tool1:
      path: "/usr/bin/tool1"
      args: ["-arg1"]
operations:
  operations:
    first_stage: ["op1", "op2"]
    second_stage: ["op3", "op4"]
nightmode_ops: ["op5"]
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    # Act
    config = Config(path=config_file, verbose=True)

    # Assert
    assert config.models == {"model1": {"path": "path/to/model1", "default": True}}


def test_config_model_path_property(tmp_path):
    # Arrange
    config_content = """
models:
  model1:
    path: "path/to/model1"
    default: true
tools:
  posix:
    tool1:
      path: "/usr/bin/tool1"
      args: ["-arg1"]
operations:
  operations:
    first_stage: ["op1", "op2"]
    second_stage: ["op3", "op4"]
nightmode_ops: ["op5"]
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    # Act
    config = Config(path=config_file, verbose=True)

    # Assert
    assert config.model_path == Path("path/to/model1")


def test_config_operations_property(tmp_path):
    # Arrange
    config_content = """
models:
  model1:
    path: "path/to/model1"
    default: true
tools:
  posix:
    tool1:
      path: "/usr/bin/tool1"
      args: ["-arg1"]
operations:
  operations:
    first_stage: ["op1", "op2"]
    second_stage: ["op3", "op4"]
nightmode_ops: ["op5"]
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    # Act
    config = Config(path=config_file, verbose=True)
    operations = config.operations

    # Assert
    assert "operations" in operations
    assert "first_stage" in operations["operations"]
    assert "second_stage" in operations["operations"]


def test_config_nightmode_operations(tmp_path):
    # Arrange
    config_content = """
models:
  model1:
    path: "path/to/model1"
    default: true
tools:
  posix:
    tool1:
      path: "/usr/bin/tool1"
      args: ["-arg1"]
operations:
  operations:
    first_stage: ["op1", "op2"]
    second_stage: ["op3", "op4"]
nightmode_ops: ["op5"]
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    # Act
    config = Config(path=config_file, verbose=True)
    operations = config.operations

    # Assert
    assert "op5" in operations["operations"]["first_stage"]
    assert "op5" not in operations["operations"]["second_stage"]


def test_config_tools_property(tmp_path):
    # Arrange
    config_content = """
models:
  model1:
    path: "path/to/model1"
    default: true
tools:
  posix:
    tool1:
      path: "/usr/bin/tool1"
      args: ["-arg1"]
operations:
  operations:
    first_stage: ["op1", "op2"]
    second_stage: ["op3", "op4"]
nightmode_ops: ["op5"]
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    # Act
    config = Config(path=config_file, verbose=True)
    tools = config.tools

    # Assert
    assert hasattr(tools, "_tool1")


from pathlib import Path

from src.nind_denoise.config.config import Config


def test_tool_initialization_with_valid_path(tmp_path):
    from src.nind_denoise.config.config import Tool

    tool_path = tmp_path / "fake_tool"
    tool_path.touch()
    args = ["-arg1"]
    tool = Tool(path=tool_path, args=args)
    assert tool.path == tool_path
    assert tool.args == args


def test_tool_initialization_with_invalid_path():
    from src.nind_denoise.config.config import Tool

    invalid_path = Path("/nonexistent/path")
    args = ["-arg1"]
    try:
        Tool(path=invalid_path, args=args)
    except ExternalToolNotFound as e:
        assert str(e) == f"External tool not found at path: {invalid_path}"


def test_tool_args_property():
    from src.nind_denoise.config.config import Tool

    tmp_path = Path("/tmp")
    args = ["-arg1"]
    tool = Tool(path=tmp_path, args=args)
    assert tool.args == args


def test_tool_append_arg():
    from src.nind_denoise.config.config import Tool

    tmp_path = Path("/tmp")
    args = ["-arg1"]
    tool = Tool(path=tmp_path, args=args)
    tool.append_arg("-arg2")
    assert tool.args == ["-arg1", "-arg2"]


def test_tools_platform_key():
    from src.nind_denoise.config.config import Tools

    platform_key = Tools._get_platform_key()
    if "windows" in platform.system().lower():
        assert platform_key == "windows"
    else:
        assert platform_key == "posix"


def test_tools_initialization(tmp_path):
    from src.nind_denoise.config.config import Tools

    tools_cfg = {
        "posix": {"tool1": {"path": str(tmp_path / "fake_tool"), "args": ["-arg1"]}}
    }
    tool_path = tmp_path / "fake_tool"
    tool_path.touch()
    tools = Tools(tools_cfg=tools_cfg)
    assert hasattr(tools, "_tool1")
