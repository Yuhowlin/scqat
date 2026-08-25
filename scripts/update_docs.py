"""Regenerate the DERIVED blocks in CLAUDE.md from the code they describe.

Two lists in CLAUDE.md are facts about the tree, and both had rotted by v0.20.0:

* the tool -> consuming-estimator table, which drives the blast-radius rule for a
  `tools/` edit. It listed 10 tools; 25 have consumers. `iq_reduce` showed 5
  consumers against a real 8, `peak_fit` 5 against 7, `hankel_analysis` named a
  module that does not exist (it is `hankel.py`), and `lockin` + `fit_stretched_exp`
  - both wired in at v0.20.0 - were missing entirely. An under-listed row means a
  `tools/` edit ships with an under-run test set.
* the estimator coverage gap, which read "9 of 27" when the tree held 43.

    python scripts/update_docs.py            # rewrite the blocks in place
    python scripts/update_docs.py --check    # exit 1 if stale (CI)

`tests/test_docs_current.py` runs the --check form.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
ESTIMATORS = REPO_ROOT / "scqat" / "estimators"
TOOLS = REPO_ROOT / "scqat" / "tools"
TESTS = REPO_ROOT / "tests"

BLOCKS = {
    "tool-consumers": "<!-- {} generated: tool-consumers -->",
    "coverage-gap": "<!-- {} generated: coverage-gap -->",
}


def estimator_names() -> list[str]:
    """Every estimator subpackage (the layout rule: one per experiment, always a package)."""
    return sorted(
        d.name for d in ESTIMATORS.iterdir() if d.is_dir() and (d / "__init__.py").is_file()
    )


def tool_names() -> list[str]:
    return sorted(p.stem for p in TOOLS.glob("*.py") if not p.stem.startswith("_"))


def exported_classes() -> dict[str, list[str]]:
    """module name -> the classes estimators/__init__.py re-exports from it (via AST)."""
    tree = ast.parse((ESTIMATORS / "__init__.py").read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module.split(".")[-1]
            out.setdefault(mod, []).extend(a.name for a in node.names)
    return out


def _tests_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in TESTS.glob("*.py"))


def tool_consumers() -> dict[str, list[str]]:
    """tool -> the estimator families that import it."""
    tools = tool_names()
    found: dict[str, set[str]] = {t: set() for t in tools}
    for f in ESTIMATORS.rglob("*.py"):
        family = f.relative_to(ESTIMATORS).parts[0]
        if family.endswith(".py"):
            family = family[:-3]
        text = f.read_text(encoding="utf-8", errors="ignore")
        for t in tools:
            if re.search(rf"\btools\.{re.escape(t)}\b|\btools import [^\n]*\b{re.escape(t)}\b", text):
                found[t].add(family)
    return {t: sorted(v) for t, v in found.items()}


def untested_estimators() -> list[str]:
    """Estimators no test imports, by module path OR by exported class name."""
    tests, classes = _tests_text(), exported_classes()
    out = []
    for name in estimator_names():
        by_path = re.search(rf"estimators\.{re.escape(name)}\b", tests)
        by_class = any(
            re.search(rf"\b{re.escape(c)}\b", tests) for c in classes.get(name, []) if c.isidentifier()
        )
        if not (by_path or by_class):
            out.append(name)
    return out


def render_tool_consumers() -> str:
    consumers = tool_consumers()
    used = {t: c for t, c in consumers.items() if c}
    unused = [t for t, c in consumers.items() if not c]
    rows = [f"| `{t}` | {', '.join(c)} |" for t, c in sorted(used.items())]
    return "\n".join(
        [
            BLOCKS["tool-consumers"].format("BEGIN"),
            "**GENERATED** - refresh with `python scripts/update_docs.py`. A `tools/` edit reaches",
            "every family listed beside it; run those families' test files too.",
            "",
            "| tool | consuming estimator families |",
            "|---|---|",
            *rows,
            "",
            "No estimator imports these (shared machinery, workflow-only, or a fitter reached through",
            f"the `get_fitter()` factory): {', '.join('`' + t + '`' for t in unused)}.",
            BLOCKS["tool-consumers"].format("END"),
        ]
    )


def render_coverage_gap() -> str:
    missing, total = untested_estimators(), len(estimator_names())
    return "\n".join(
        [
            BLOCKS["coverage-gap"].format("BEGIN"),
            f"**GENERATED** - refresh with `python scripts/update_docs.py`. **{len(missing)} of {total}**",
            "estimators are imported by NO test, by module path or exported class:",
            "",
            *[f"- `{m}`" for m in missing],
            BLOCKS["coverage-gap"].format("END"),
        ]
    )


RENDERERS = {"tool-consumers": render_tool_consumers, "coverage-gap": render_coverage_gap}


def current_block(text: str, key: str) -> str:
    begin, end = BLOCKS[key].format("BEGIN"), BLOCKS[key].format("END")
    return text[text.index(begin) : text.index(end) + len(end)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if stale instead of rewriting")
    args = ap.parse_args()

    text = CLAUDE_MD.read_text(encoding="utf-8")
    stale = []
    for key, render in RENDERERS.items():
        begin, end = BLOCKS[key].format("BEGIN"), BLOCKS[key].format("END")
        if begin not in text or end not in text:
            print(f"CLAUDE.md: missing the {key} markers", file=sys.stderr)
            return 2
        have, wanted = current_block(text, key), render()
        if have != wanted:
            stale.append(key)
            text = text.replace(have, wanted)

    if not stale:
        print("CLAUDE.md: generated blocks are current")
        return 0
    if args.check:
        print(
            f"CLAUDE.md: STALE generated block(s): {', '.join(stale)}.\n"
            f"Run `python scripts/update_docs.py` and commit the result.",
            file=sys.stderr,
        )
        return 1
    CLAUDE_MD.write_text(text, encoding="utf-8", newline="\n")
    print(f"CLAUDE.md: rewrote {', '.join(stale)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
