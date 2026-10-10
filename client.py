# SPDX-License-Identifier: BSD-3-Clause

"""Verified Compute Lab client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ComputeLabAction, ComputeLabObservation


class ComputeLabEnv(EnvClient[ComputeLabAction, ComputeLabObservation, State]):
    """
    WebSocket client for the Verified Compute Lab environment.

    Examples:

    ```python
    with ComputeLabEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(task_id="recurrence_mod-l2")
        result = env.step(ComputeLabAction(tool="read", path="sequence.txt"))
    ```
    """

    def _step_payload(self, action: ComputeLabAction) -> Dict:
        return action.model_dump(exclude_none=True, exclude={"metadata"})

    def _parse_result(self, payload: Dict) -> StepResult[ComputeLabObservation]:
        obs = payload.get("observation", {})
        observation = ComputeLabObservation(
            output=obs.get("output", ""),
            steps_left=obs.get("steps_left", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(**payload)
