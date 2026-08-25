"""CLAUDE.md's derived blocks must match the tree they describe.

Two of its lists are facts about the code, and both had rotted by v0.20.0: the
tool -> consumer table (10 tools listed, 25 have consumers; `iq_reduce` showed 5
consumers against a real 8) and the estimator coverage gap ("9 of 27" when the
tree held 43). The first one matters most - it is what a `tools/` edit consults
to decide which families to re-test, so an under-listed row ships an under-run
test set.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "update_docs.py"


@pytest.fixture(scope="module")
def update_docs():
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    spec = importlib.util.spec_from_file_location("update_docs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["update_docs"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("block", ["tool-consumers", "coverage-gap"])
def test_generated_block_is_current(update_docs, block):
    text = update_docs.CLAUDE_MD.read_text(encoding="utf-8")
    wanted = update_docs.RENDERERS[block]()
    assert update_docs.current_block(text, block) == wanted, (
        f"CLAUDE.md's {block!r} block is stale.\n"
        "Run `python scripts/update_docs.py` and commit the result."
    )


def test_every_tool_with_consumers_is_listed(update_docs):
    """The invariant behind the table, independent of its formatting."""
    text = update_docs.CLAUDE_MD.read_text(encoding="utf-8")
    missing = [t for t, c in update_docs.tool_consumers().items() if c and f"`{t}`" not in text]
    assert not missing, f"tools with estimator consumers but absent from CLAUDE.md: {missing}"
