# SPDX-License-Identifier: BSD-3-Clause

"""Data models for the Verified Compute Lab environment."""

from typing import Any, Dict, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ComputeLabAction(Action):
    """One tool call: read a task file, run Python on the files, or submit the answer."""

    tool: str = Field(..., description='One of "read", "python" or "submit".')
    path: Optional[str] = Field(
        default=None, description='File name to show (tool "read").'
    )
    code: Optional[str] = Field(
        default=None,
        description='Python 3 source to run in the task folder; print what you need (tool "python").',
    )
    answer: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Final answer, one key per answer field (tool "submit"). Ends the episode.',
    )


class ComputeLabObservation(Observation):
    """Tool output, or the task statement on reset."""

    output: str = Field(default="", description="Task statement or tool output")
    steps_left: int = Field(default=0, description="Tool calls left in this episode")
