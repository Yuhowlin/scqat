# Environments

Which Python environment to use lives in one place for all four repos, because the rule is a
**combo** property — the vendor-version half of it is literally cross-repo, and separate copies
disagree within one cycle:

**https://github.com/shiau109/SCQO/blob/main/ENVIRONMENTS.md**

The one line for this repo, so you cannot get it wrong by not clicking:

```bash
uv run --extra dev pytest tests/test_<name>_estimator.py -q
```

**`--extra dev` is mandatory** — `pytest` is in `[project.optional-dependencies]`, not
`[dependency-groups]`, and uv installs only the latter by default; bare `uv run pytest` dies with
`Failed to spawn: pytest`. That runs in `scqat/.venv`, uv's project env, which `uv run` creates
and keeps in step on every invocation. scqat needs no sibling checkout to test itself.

This repo runs on the same **3.10–3.12** window as the rest of the combo.
