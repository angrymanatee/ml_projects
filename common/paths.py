import subprocess
from pathlib import Path


def get_repo_root() -> Path:
    """Get the repo root directory, expecting this to be a git repository."""
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode()
        .strip()
    )


def get_data_dir() -> Path:
    return get_repo_root() / "data"
