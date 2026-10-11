# SPDX-License-Identifier: BSD-3-Clause

"""Data models for the Verified Compute Lab (Workbench v2) environment."""

from typing import Any, Dict

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ComputeLabAction(Action):
    """One tool call. The first observation lists the tools this task offers and their args."""

    tool: str = Field(..., description='Tool name, e.g. "read" or "submit"; see the task\'s tool list.')
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description='Tool arguments as an object, e.g. {"path": "notes.md"}; {"answer": {...}} for "submit".',
    )


class ComputeLabObservation(Observation):
    """Tool output, or the task statement on reset."""

    output: str = Field(default="", description="Task statement or tool output")
    steps_left: int = Field(default=0, description="Tool calls left in this episode")
