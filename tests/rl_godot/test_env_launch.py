from rl_godot.env_launch import build_launch_cmd


def test_no_user_args_omits_separator() -> None:
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


def test_map_appended_after_separator() -> None:
    cmd = build_launch_cmd(
        "/path/to/MazeBots.app",
        port=11008,
        seed=0,
        show_window=False,
        framerate=None,
        action_repeat=None,
        speedup=None,
        rl_config=None,
        map_name="GoStraightTrap",
    )
    tokens = cmd.split(" ")
    separator_index = tokens.index("--")
    assert tokens[separator_index + 1] == "--map=GoStraightTrap"


def test_map_and_rl_config_share_one_separator() -> None:
    cmd = build_launch_cmd(
        "/path/to/MazeBots.app",
        port=11008,
        seed=0,
        show_window=False,
        framerate=None,
        action_repeat=None,
        speedup=None,
        rl_config="no_rays",
        map_name="GoStraightTrap",
    )
    tokens = cmd.split(" ")
    assert tokens.count("--") == 1
    separator_index = tokens.index("--")
    assert tokens[separator_index + 1 :] == [
        "--rl-config=res://configs/no_rays.cfg",
        "--map=GoStraightTrap",
    ]


def test_res_path_map_passed_through_verbatim() -> None:
    # MapSelection.ResolvePath accepts a full scene path as well as a bare name.
    cmd = build_launch_cmd(
        "/path/to/MazeBots.app",
        port=11008,
        seed=0,
        show_window=False,
        framerate=None,
        action_repeat=None,
        speedup=None,
        rl_config=None,
        map_name="res://maps/GoStraightTrap.tscn",
    )
    assert cmd.endswith("-- --map=res://maps/GoStraightTrap.tscn")


def test_layout_config_and_seed_share_the_separator() -> None:
    cmd = build_launch_cmd(
        "/path/to/MazeBots.app",
        port=11008,
        seed=0,
        show_window=False,
        framerate=None,
        action_repeat=None,
        speedup=None,
        rl_config="watch",
        map_name="GoStraightTrap",
        layout_config="layout_GoStraightTrap",
        layout_seed=99,
    )
    tokens = cmd.split(" ")
    assert tokens.count("--") == 1
    separator_index = tokens.index("--")
    assert tokens[separator_index + 1 :] == [
        "--rl-config=res://configs/watch.cfg",
        "--map=GoStraightTrap",
        "--layout-config=res://configs/layout_GoStraightTrap.cfg",
        "--layout-seed=99",
    ]


def test_layout_args_omitted_when_unset() -> None:
    cmd = build_launch_cmd(
        "/path/to/MazeBots.app",
        port=11008,
        seed=0,
        show_window=False,
        framerate=None,
        action_repeat=None,
        speedup=None,
        rl_config=None,
        map_name=None,
    )
    assert "--" not in cmd.split(" ")
    assert "layout" not in cmd
