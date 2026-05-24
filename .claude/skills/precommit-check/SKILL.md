---
name: precommit-check
description: Use when about to commit changes, after finishing a task, or when asked to verify code quality. Runs formatting, linting, type checking, and tests.
---

# Pre-Commit Check

Run this before every commit:

```bash
./scripts/check.sh
```

This runs in order:
1. **black** — reformats `.py` and `.ipynb` files in place
2. **ruff** — fixes safe lint issues, fails on remaining errors
3. **pytest** — runs all tests with coverage
4. **pre-commit** — full hook suite (black, ruff, pyright, nbstripout)

If anything fails, fix it and re-run the script. Black and ruff auto-fix what they can; stage those changes before committing.
