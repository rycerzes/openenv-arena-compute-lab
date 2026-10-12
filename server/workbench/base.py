# SPDX-License-Identifier: BSD-3-Clause

"""Workbench v2 building blocks: tools, episodes and task families.

An episode is a generated task with its own tool set. The agent's action is
always `{"tool": <name>, "args": {...}}`; the first observation describes the
tools available for this task, so the policy has to read each interface.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from ..sandbox import Workspace
from ..tasks import grade as grade_fields
from ..tasks.base import AnswerField

READ_MAX_LINES = 30
READ_MAX_CHARS = 1800


class ToolError(Exception):
    """A tool call with bad arguments; reported to the agent without ending the episode."""


@dataclass
class Tool:
    """A callable the agent can use: `usage` documents its args in the prompt."""

    name: str
    usage: str
    run: Callable[[dict[str, Any]], str]


@dataclass
class Episode:
    """A generated task instance: what the agent sees, its tools, and how it is graded."""

    question: str
    tools: list[Tool]
    answer_hint: str
    grade: Callable[[Any], tuple[float, str]]
    workspace: Workspace | None = None
    oracle: list[dict[str, Any]] = field(default_factory=list)  # actions that score 1.0; never shown to the agent
    extra_cleanup: list[Callable[[], None]] = field(default_factory=list)

    def close(self) -> None:
        if self.workspace is not None:
            self.workspace.close()
        for fn in self.extra_cleanup:
            fn()


@dataclass(frozen=True)
class Family:
    """A task family; `make(difficulty, rng)` builds an episode at difficulty 1-4."""

    name: str
    domain: str
    summary: str
    make: Callable[[int, Any], Episode]


def arg(args: dict[str, Any], name: str, kind: type = str, required: bool = True) -> Any:
    value = args.get(name)
    if value is None:
        if required:
            raise ToolError(f'missing argument "{name}"')
        return None
    if kind is int and isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value)
    if kind is int and isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
        raise ToolError(f'argument "{name}" must be {kind.__name__}')
    return value


def read_text(text: str, start: int = 1, max_lines: int = READ_MAX_LINES) -> str:
    """Whole lines from `start`, stopping at the line or character budget, never cutting mid-output."""
    lines = text.splitlines()
    start = max(1, start)
    chunk, size = [], 0
    for line in lines[start - 1 : start - 1 + max_lines]:
        line = line if len(line) <= 300 else line[:300] + " [...]"
        if chunk and size + len(line) + 1 > READ_MAX_CHARS:
            break
        chunk.append(line)
        size += len(line) + 1
    out = "\n".join(chunk)
    rest = len(lines) - (start - 1 + len(chunk))
    if rest > 0:
        out += f"\n[{rest} more lines; pass start={start + len(chunk)} or process the file with code]"
    return out


def file_tools(ws: Workspace, files: dict[str, str], python: bool = True) -> list[Tool]:
    """list/read (and optionally python) over a workspace of text files."""

    def do_list(args):
        return "\n".join(f"{name} ({len(files[name].splitlines())} lines)" for name in files)

    def do_read(args):
        name = os.path.basename(arg(args, "path").strip())
        if name not in files:
            raise ToolError(f"no file {name!r}; files: {', '.join(files)}")
        start = arg(args, "start", int, required=False) or 1
        return read_text(files[name], start)

    def do_python(args):
        code = arg(args, "code")
        if not code.strip():
            raise ToolError('"code" is empty')
        return ws.run_python(code)

    tools = [
        Tool("list", '{"tool": "list", "args": {}}  list the task files', do_list),
        Tool("read", '{"tool": "read", "args": {"path": "<file>", "start": 1}}  show 30 lines from line start', do_read),
    ]
    if python:
        tools.append(Tool(
            "python",
            '{"tool": "python", "args": {"code": "<Python 3 standard library>"}}  runs in the folder with the files; print results',
            do_python,
        ))
    return tools


def fields_grader(fields: list[AnswerField]) -> Callable[[Any], tuple[float, str]]:
    """All-or-nothing: partial credit would reward guessing categorical fields without doing the work."""

    def grade(answer):
        if not isinstance(answer, dict):
            return 0.0, "answer must be an object with one key per answer field"
        _, wrong = grade_fields(fields, answer)
        detail = f" Wrong: {', '.join(wrong)}." if wrong else ""
        return (0.0 if wrong else 1.0), f"{len(fields) - len(wrong)}/{len(fields)} fields correct.{detail}"

    return grade


def submit_action(answer: Any) -> dict[str, Any]:
    return {"tool": "submit", "args": {"answer": answer}}


def submit_hint(example: dict[str, Any]) -> str:
    """The prompt line showing the submit call, with `example` as the answer."""
    return f"Submit: {json.dumps(submit_action(example))}"


def fields_hint(fields: list[AnswerField]) -> str:
    lines = "\n".join(f"- {f.name}: {f.description}" for f in fields)
    return f"Answer fields:\n{lines}\n" + submit_hint({f.name: "..." for f in fields})


def fields_oracle(fields: list[AnswerField]) -> list[dict[str, Any]]:
    return [submit_action({f.name: f.expected for f in fields})]
