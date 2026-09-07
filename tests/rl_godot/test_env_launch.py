import signal
import subprocess
from typing import cast

import pytest

import rl_godot.env_launch as env_launch
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


class _FakeProc:
    """Stand-in for `subprocess.Popen` that "dies" only when it receives `dies_on`.

    `os.killpg` is mocked to call `receive()` directly rather than actually
    signaling anything, so these tests never touch a real process.
    """

    def __init__(self, pid: int, dies_on: int | None = signal.SIGTERM) -> None:
        self.pid = pid
        self.dies_on = dies_on
        self._alive = True

    def poll(self) -> int | None:
        return None if self._alive else 0

    def wait(self, timeout: float | None = None) -> int:
        if not self._alive:
            return 0
        raise subprocess.TimeoutExpired(cmd="godot", timeout=timeout or 0)

    def receive(self, sig: int) -> None:
        if sig == self.dies_on:
            self._alive = False


def test_cleanup_terminates_registered_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _FakeProc(pid=4242, dies_on=signal.SIGTERM)
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(env_launch.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        env_launch.os,
        "killpg",
        lambda pgid, sig: (calls.append((pgid, sig)), proc.receive(sig)),
    )
    env_launch._LAUNCHED_PROCESSES.append(cast(subprocess.Popen, proc))

    env_launch._cleanup_launched_processes()

    assert calls == [(4242, signal.SIGTERM)]
    assert proc.poll() == 0
    assert env_launch._LAUNCHED_PROCESSES == []


def test_cleanup_escalates_to_sigkill_when_terminate_does_not_stop_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _FakeProc(pid=99, dies_on=signal.SIGKILL)
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(env_launch.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        env_launch.os,
        "killpg",
        lambda pgid, sig: (calls.append((pgid, sig)), proc.receive(sig)),
    )
    monkeypatch.setattr(env_launch, "_TERMINATE_TIMEOUT_SECONDS", 0.01)
    env_launch._LAUNCHED_PROCESSES.append(cast(subprocess.Popen, proc))

    env_launch._cleanup_launched_processes()

    assert calls == [(99, signal.SIGTERM), (99, signal.SIGKILL)]


def test_cleanup_skips_process_that_already_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # e.g. GodotEnv.close()'s graceful "close" message already made it exit.
    proc = _FakeProc(pid=123)
    proc._alive = False
    calls: list[tuple[int, int]] = []

    def _unexpected_getpgid(pid: int) -> int:
        raise AssertionError("getpgid should not be called on an exited process")

    monkeypatch.setattr(env_launch.os, "getpgid", _unexpected_getpgid)
    monkeypatch.setattr(
        env_launch.os, "killpg", lambda pgid, sig: calls.append((pgid, sig))
    )
    env_launch._LAUNCHED_PROCESSES.append(cast(subprocess.Popen, proc))

    env_launch._cleanup_launched_processes()

    assert calls == []
    assert env_launch._LAUNCHED_PROCESSES == []
