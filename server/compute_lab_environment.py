# SPDX-License-Identifier: BSD-3-Clause

"""Verified Compute Lab environment (Workbench v2).

Each reset builds a fresh task from one of 14 families across eight domains.
Every task has its own tools, listed in the first observation; the action is
always `{"tool": <name>, "args": {...}}`. Submitting ends the episode and the
family's grader returns a reward in [0, 1].
"""

import time
import zlib
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ComputeLabAction, ComputeLabObservation
    from .workbench import TASK_IDS, Episode, make_episode, parse_task_id
    from .workbench.base import Tool, ToolError
except ImportError:
    from models import ComputeLabAction, ComputeLabObservation
    from server.workbench import TASK_IDS, Episode, make_episode, parse_task_id
    from server.workbench.base import Tool, ToolError

MAX_TOOL_CALLS = 12
# Without an explicit seed, resets of the same task within one window share an instance. The
# rollouts of a GRPO group reset together, so their rewards compare the policy on one task
# instead of mixing in how hard each random draw was; later steps still get fresh instances.
GROUP_WINDOW_S = 600


def group_seed(task_id: str, now: Optional[float] = None) -> int:
    window = int((time.time() if now is None else now) // GROUP_WINDOW_S)
    return zlib.crc32(f"{task_id}:{window}".encode())
DEFAULT_TASK_ID = "code_repair"


def render_prompt(episode: Episode) -> str:
    tools = "\n".join(t.usage for t in episode.tools)
    return (
        f"{episode.question}\n"
        "Reply with one JSON object and nothing else: no prose, no code fences. Inside strings write "
        'newlines as \\n and quotes as \\". Keep code short. Tools:\n'
        f"{tools}\n"
        '{"tool": "submit", "args": {"answer": ...}}  ends the episode and grades the answer\n'
        f"{episode.answer_hint}\n"
        f"You have {MAX_TOOL_CALLS} tool calls."
    )


class ComputeLabEnvironment(Environment):
    """Procedural tool-use tasks across eight domains with sandboxed tools."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode: Optional[Episode] = None
        self._tools: dict[str, Tool] = {}
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
        parse_task_id(task_id)  # reject a bad task_id before closing the current episode
        if seed is None:
            seed = group_seed(task_id)
        self.close()
        self._episode, difficulty = make_episode(task_id, seed)
        self._tools = {t.name: t for t in self._episode.tools}
        self._steps_left = MAX_TOOL_CALLS
        self._done = False
        self._reward = 0.0
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            seed=seed,
            difficulty=difficulty,
        )
        return self._observe(render_prompt(self._episode))

    def step(self, action: ComputeLabAction, timeout_s: Optional[float] = None, **kwargs: Any) -> ComputeLabObservation:  # type: ignore[override]
        if self._done or self._episode is None:
            return self._observe("The episode is over; call reset.")
        self._state.step_count += 1
        self._steps_left -= 1
        name = (action.tool or "").strip().lower()
        args = action.args if isinstance(action.args, dict) else {}
        if name == "submit":
            if "answer" not in args:
                return self._after_tool('submit needs args {"answer": ...}')
            reward, detail = self._episode.grade(args["answer"])
            self._done, self._reward = True, float(reward)
            return self._observe(f"Graded: {detail}")
        tool = self._tools.get(name)
        if tool is None:
            return self._after_tool(f"Unknown tool {action.tool!r}; tools: {', '.join(self._tools)}, submit")
        try:
            output = tool.run(args)
        except ToolError as e:
            output = f"Error: {e}"
        return self._after_tool(output)

    def _after_tool(self, output: str) -> ComputeLabObservation:
        if self._steps_left <= 0:
            self._done, self._reward = True, 0.0
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
        if self._episode is not None:
            self._episode.close()
            self._episode = None


__all__ = ["ComputeLabEnvironment", "TASK_IDS"]
