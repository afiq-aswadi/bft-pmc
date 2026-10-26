"""The documentation and scripts stay in step with the code.

Every train.py / eval.py command a document or script names must exist, the
README must list every command, and every relative link in the Markdown files
must point at a file in the repository.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bft_pmc import cli

ROOT = Path(__file__).resolve().parents[1]
# Historical records kept unchanged on purpose; they name the old layout.
HISTORICAL = (ROOT / "slurm" / "rebuttal",)

MARKDOWN = sorted(
    path
    for path in ROOT.rglob("*.md")
    if not any(part in {".venv", "outputs", ".git"} for part in path.parts)
    and not path.is_relative_to(ROOT / "pfn_transformers")
)
SCRIPTS = sorted(
    path
    for pattern in (
        "scripts/**/*.sh",
        "slurm/*.sh",
        "slurm/*.slurm",
        "configs/**/*.yaml",
    )
    for path in ROOT.glob(pattern)
)
COMMANDS = {
    "eval": {c.name for s in cli.EVAL_SECTIONS for c in s.commands},
    "train": {c.name for s in cli.TRAIN_SECTIONS for c in s.commands},
}
# An invocation: after a backtick, `uv run` or `python`, on one line.
INVOCATION = re.compile(
    r"(?:^|`|\(|uv run |python3? )(eval|train)\.py[ \t]+([a-z][a-z0-9-]*)", re.M
)
FENCE = re.compile(r"```.*?```", re.S)
INLINE = re.compile(r"`[^`\n]+`")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _code_spans(markdown: str) -> str:
    """The text inside fenced blocks and inline code, where commands appear."""
    fenced = FENCE.findall(markdown)
    rest = FENCE.sub("", markdown)
    return "\n".join(fenced + INLINE.findall(rest))


def _named_commands(path: Path) -> list[tuple[str, str]]:
    text = path.read_text()
    if path.suffix == ".md":
        text = _code_spans(text)
    return INVOCATION.findall(text)


@pytest.mark.parametrize(
    "path", [*MARKDOWN, *SCRIPTS], ids=lambda p: str(p.relative_to(ROOT))
)
def test_named_commands_exist(path: Path) -> None:
    if path.is_relative_to(HISTORICAL[0]):
        pytest.skip("historical record")
    unknown = [
        f"{script}.py {name}"
        for script, name in _named_commands(path)
        if name not in COMMANDS[script]
    ]
    assert not unknown, f"{path.relative_to(ROOT)} names unknown commands: {unknown}"


def test_readme_lists_every_command() -> None:
    readme = (ROOT / "README.md").read_text()
    missing = sorted(
        f"{script}.py {name}"
        for script, names in COMMANDS.items()
        for name in names
        if f"`{name}`" not in readme
    )
    assert not missing


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(path: Path) -> None:
    broken = []
    for target in LINK.findall(path.read_text()):
        if re.match(r"[a-z]+://|mailto:|#", target):
            continue
        file_part = target.split("#", 1)[0]
        if not (path.parent / file_part).exists():
            broken.append(target)
    assert not broken, f"{path.relative_to(ROOT)} links to missing files: {broken}"
