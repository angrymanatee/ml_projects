import signal
import subprocess
from unittest.mock import patch

from rl_godot.kill_strays import _is_stray_command, find_strays, kill_strays


def test_is_stray_command_matches_exported_binary() -> None:
    command = (
        "/Users/sauron/GodotProjects/maze_bots/build/macos/MazeBots.app/Contents/"
        "MacOS/MazeBots --port=11026 --env_seed=18 --disable-render-loop --headless"
    )
    assert _is_stray_command(command)


def test_is_stray_command_rejects_editor() -> None:
    command = "/Applications/Godot_mono.app/Contents/MacOS/Godot --path /some/project"
    assert not _is_stray_command(command)


def test_is_stray_command_rejects_editor_even_with_matching_flags() -> None:
    # The editor never actually carries these flags, but the exclusion is by
    # bundle name, not by flag absence -- this should hold either way.
    command = (
        "/Applications/Godot_mono.app/Contents/MacOS/Godot --port=11026 --env_seed=18"
    )
    assert not _is_stray_command(command)


def test_is_stray_command_requires_both_flags() -> None:
    assert not _is_stray_command("/path/to/MazeBots --port=11026")
    assert not _is_stray_command("/path/to/MazeBots --env_seed=18")
    assert not _is_stray_command("/path/to/MazeBots")


def test_find_strays_parses_ps_output_and_filters() -> None:
    ps_output = (
        "  PID COMMAND\n"
        "  111 /Applications/Godot_mono.app/Contents/MacOS/Godot --path /proj\n"
        "  222 /path/to/MazeBots --port=11026 --env_seed=18 --headless\n"
        "  333 /usr/bin/some-other-process --port=1234\n"
    )
    with patch(
        "rl_godot.kill_strays.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout=ps_output
        ),
    ) as mock_run:
        strays = find_strays()

    assert strays == [(222, "/path/to/MazeBots --port=11026 --env_seed=18 --headless")]
    assert mock_run.call_args.kwargs["check"] is True


def test_kill_strays_sends_sigterm_then_sigkill_to_survivors() -> None:
    calls: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        if sig == 0:  # liveness probe
            if pid == 2:
                return  # pid 2 survived SIGTERM
            raise ProcessLookupError  # pid 1 exited after SIGTERM
        calls.append((pid, sig))

    with (
        patch("rl_godot.kill_strays.os.kill", side_effect=fake_kill),
        patch("rl_godot.kill_strays.time.sleep") as mock_sleep,
    ):
        kill_strays([(1, "cmd1"), (2, "cmd2")], grace_period=5.0)

    assert (1, signal.SIGTERM) in calls
    assert (2, signal.SIGTERM) in calls
    assert (2, signal.SIGKILL) in calls
    assert (1, signal.SIGKILL) not in calls
    mock_sleep.assert_called_once_with(5.0)


def test_kill_strays_skips_sleep_when_nothing_to_kill() -> None:
    with (
        patch("rl_godot.kill_strays.os.kill") as mock_kill,
        patch("rl_godot.kill_strays.time.sleep") as mock_sleep,
    ):
        kill_strays([], grace_period=5.0)

    mock_kill.assert_not_called()
    mock_sleep.assert_not_called()
