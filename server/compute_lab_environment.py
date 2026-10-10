# SPDX-License-Identifier: BSD-3-Clause

"""Verified Compute Lab environment.

Each reset generates a fresh task instance (or a reproducible one when a seed
is given): a few small files, a question and named answer fields. The agent
reads files, runs Python on them and submits; the verifier compares every
field against an exact recomputation and the reward is the share of correct
fields.
"""

import json
import os
import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ComputeLabAction, ComputeLabObservation
    from .sandbox import Workspace, clip
    from .tasks import TASK_IDS, TaskInstance, generate, grade, parse_task_id
except ImportError:
    from models import ComputeLabAction, ComputeLabObservation
    from server.sandbox import Workspace, clip
    from server.tasks import TASK_IDS, TaskInstance, generate, grade, parse_task_id

MAX_TOOL_CALLS = 12
DEFAULT_TASK_ID = "loan_schedule-l1"
READ_MAX_LINES = 40
READ_MAX_CHARS = 2500


def render_prompt(task: TaskInstance) -> str:
    example = {f.name: "..." for f in task.fields}
    fields = "\n".join(f"- {f.name}: {f.description}" for f in task.fields)
    return (
        f"{task.question}\n"
        f"Files: {', '.join(task.files)}\n"
        f"Answer fields:\n{fields}\n"
        "Reply with exactly one JSON object per turn, one of:\n"
        '{"tool": "read", "path": "<file>"}\n'
        '{"tool": "python", "code": "<Python 3 standard library; runs in the folder with the files; print the results>"}\n'
        f'{{"tool": "submit", "answer": {json.dumps(example)}}}\n'
        f"You have {MAX_TOOL_CALLS} tool calls. Submit ends the episode; every field is checked "
        "against an exact recomputation."
    )


class ComputeLabEnvironment(Environment):
    """Procedural math, science and finance tasks with a sandboxed Python tool."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task: Optional[TaskInstance] = None
        self._workspace: Optional[Workspace] = None
        self._steps_left = 0
        self._done = True
        self._reward = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ComputeLabObservation:
        task_id = task_id or DEFAULT_TASK_ID
        parse_task_id(task_id)
        if seed is None:
            seed = random.SystemRandom().randrange(2**31)
        self.close()
        self._task = generate(task_id, seed)
        self._workspace = Workspace(self._task.files)
        self._steps_left = MAX_TOOL_CALLS
        self._done = False
        self._reward = 0.0
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            seed=seed,
        )
        return ComputeLabObservation(
            output=render_prompt(self._task),
            steps_left=self._steps_left,
            done=False,
            reward=0.0,
        )

    def step(self, action: ComputeLabAction, timeout_s: Optional[float] = None, **kwargs: Any) -> ComputeLabObservation:  # type: ignore[override]
        if self._done or self._task is None:
            return self._observe("The episode is over; call reset.")
        self._state.step_count += 1
        self._steps_left -= 1
        tool = (action.tool or "").strip().lower()
        if tool == "submit":
            if not isinstance(action.answer, dict):
                return self._after_tool('submit needs "answer": an object with one key per answer field.')
            reward, wrong = grade(self._task.fields, action.answer)
            self._done = True
            self._reward = reward
            right = len(self._task.fields) - len(wrong)
            detail = f" Wrong: {', '.join(wrong)}." if wrong else ""
            return self._observe(f"Graded: {right}/{len(self._task.fields)} fields correct.{detail}")
        if tool == "read":
            return self._after_tool(self._read(action.path))
        if tool == "python":
            if not action.code or not action.code.strip():
                return self._after_tool('python needs "code".')
            return self._after_tool(self._workspace.run_python(action.code))
        return self._after_tool(f'Unknown tool {action.tool!r}; use "read", "python" or "submit".')

    def _read(self, path: Optional[str]) -> str:
        name = os.path.basename((path or "").strip())
        if name not in self._task.files:
            return f"No file {path!r}. Files: {', '.join(self._task.files)}"
        lines = self._task.files[name].splitlines()
        text = "\n".join(lines[:READ_MAX_LINES])
        if len(lines) > READ_MAX_LINES:
            text += f"\n[{len(lines) - READ_MAX_LINES} more lines; process the file with python]"
        return clip(text, READ_MAX_CHARS)

    def _after_tool(self, output: str) -> ComputeLabObservation:
        if self._steps_left <= 0:
            self._done = True
            self._reward = 0.0
            return self._observe(f"{output}\nOut of tool calls without a submission; reward 0.")
        if self._steps_left == 1:
            output += "\n[1 tool call left: submit now]"
        return self._observe(output)

    def _observe(self, output: str) -> ComputeLabObservation:
        return ComputeLabObservation(
            output=output,
            steps_left=self._steps_left,
            done=self._done,
            reward=self._reward if self._done else 0.0,
        )

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        if self._workspace is not None:
            self._workspace.close()
            self._workspace = None


__all__ = ["ComputeLabEnvironment", "TASK_IDS"]
