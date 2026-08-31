from rl_godot.env_launch import build_launch_cmd


def test_no_rl_config_omits_separator() -> None:
    cmd = build_launch_cmd(
        "/path/to/MazeBots.app",
        port=11008,
        seed=0,
        show_window=False,
        framerate=None,
        action_repeat=None,
        speedup=None,
        rl_config=None,
    )
    assert "--" not in cmd.split(" ")
    assert "--port=11008" in cmd
    assert "--disable-render-loop" in cmd


def test_rl_config_appended_after_separator() -> None:
    cmd = build_launch_cmd(
        "/path/to/MazeBots.app",
        port=11008,
        seed=0,
        show_window=False,
        framerate=None,
        action_repeat=None,
        speedup=None,
        rl_config="no_rays",
    )
    tokens = cmd.split(" ")
    separator_index = tokens.index("--")
    assert tokens[separator_index + 1] == "--rl-config=res://configs/no_rays.cfg"
    assert tokens[-1] == "--rl-config=res://configs/no_rays.cfg"
