import subprocess
from pathlib import Path


def get_repo_root() -> Path:
    """Return the repo root via git, falling back to the package's parent when
    running outside a git repo (e.g. on a remote training pod)."""
    try:
        return Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return Path(__file__).parent.parent


def get_data_dir() -> Path:
    return get_repo_root() / "data"
