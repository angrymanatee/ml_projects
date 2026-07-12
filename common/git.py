import subprocess


def _run(args: list[str]) -> str:
    return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()


def get_sha(short: bool = True) -> str:
    """Return the SHA of HEAD, appending '-dirty' if there are uncommitted changes.

    Returns 'unknown' when not inside a git repo (e.g. on a remote training pod).

    Args:
        short: If True, return the abbreviated (7-char) SHA.
    """
    args = (
        ["git", "rev-parse", "--short", "HEAD"]
        if short
        else ["git", "rev-parse", "HEAD"]
    )
    try:
        sha = _run(args)
    except subprocess.CalledProcessError:
        return "unknown"
    return f"{sha}-dirty" if is_dirty() else sha


def get_branch() -> str:
    """Return the current branch name, or the short SHA if in detached HEAD state.

    Returns 'unknown' when not inside a git repo (e.g. on a remote training pod).
    """
    try:
        return _run(["git", "symbolic-ref", "--short", "HEAD"])
    except subprocess.CalledProcessError:
        pass
    try:
        return get_sha(short=True)
    except subprocess.CalledProcessError:
        return "unknown"


def is_dirty() -> bool:
    """Return True if there are uncommitted changes (tracked or staged)."""
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD"],
        stderr=subprocess.DEVNULL,
    )
    return result.returncode != 0
