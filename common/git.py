import subprocess


def _run(args: list[str]) -> str:
    return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()


def get_sha(short: bool = True) -> str:
    """Return the SHA of HEAD, appending '-dirty' if there are uncommitted changes.

    Args:
        short: If True, return the abbreviated (7-char) SHA.
    """
    args = (
        ["git", "rev-parse", "--short", "HEAD"]
        if short
        else ["git", "rev-parse", "HEAD"]
    )
    sha = _run(args)
    return f"{sha}-dirty" if is_dirty() else sha


def get_branch() -> str:
    """Return the current branch name, or the short SHA if in detached HEAD state."""
    try:
        return _run(["git", "symbolic-ref", "--short", "HEAD"])
    except subprocess.CalledProcessError:
        return get_sha(short=True)


def is_dirty() -> bool:
    """Return True if there are uncommitted changes (tracked or staged)."""
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD"],
        stderr=subprocess.DEVNULL,
    )
    return result.returncode != 0
