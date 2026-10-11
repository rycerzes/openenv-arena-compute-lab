# SPDX-License-Identifier: BSD-3-Clause

"""The v1 families (finance, science, math) behind the v2 tool interface.

Difficulty 1-3 maps to the v1 levels; difficulty 4 reuses level 3.
"""

from ..sandbox import Workspace
from ..tasks import FAMILIES as V1_FAMILIES
from .base import Episode, Family, fields_grader, fields_hint, fields_oracle, file_tools


def _adapt(v1):
    def make(difficulty, rng):
        level = min(difficulty, 3)
        task = v1.generate(level, rng)
        ws = Workspace(task.files)
        return Episode(
            question=task.question,
            tools=file_tools(ws, task.files),
            answer_hint=fields_hint(task.fields),
            grade=fields_grader(task.fields),
            workspace=ws,
            oracle=fields_oracle(task.fields),
        )

    return Family(v1.name, v1.domain, v1.summary, make)


FAMILIES = [_adapt(f) for f in V1_FAMILIES.values()]
