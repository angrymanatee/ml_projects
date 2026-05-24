from pathlib import Path

from common.paths import get_data_dir, get_repo_root


def test_get_repo_root_returns_path() -> None:
    root = get_repo_root()
    assert isinstance(root, Path)
    assert root.is_dir()


def test_get_repo_root_contains_pyproject() -> None:
    # Sanity check that we're pointing at the right repo root.
    assert (get_repo_root() / "pyproject.toml").exists()


def test_get_data_dir_is_under_repo_root() -> None:
    assert get_data_dir() == get_repo_root() / "data"


def test_get_data_dir_returns_path() -> None:
    assert isinstance(get_data_dir(), Path)
