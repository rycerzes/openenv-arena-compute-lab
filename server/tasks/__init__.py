# SPDX-License-Identifier: BSD-3-Clause

"""Task registry: task ids look like `<family>-l<level>`, e.g. `loan_schedule-l2`."""

import random

from . import finance, math_tasks, science
from .base import AnswerField, Family, TaskInstance, grade

FAMILIES: dict[str, Family] = {
    f.name: f for f in [*finance.FAMILIES, *science.FAMILIES, *math_tasks.FAMILIES]
}
LEVELS = (1, 2, 3)
TASK_IDS = [f"{name}-l{level}" for name in FAMILIES for level in LEVELS]


def parse_task_id(task_id: str) -> tuple[Family, int]:
    name, _, level = task_id.rpartition("-l")
    if name not in FAMILIES or level not in {"1", "2", "3"}:
        raise ValueError(f"unknown task_id {task_id!r}; expected one of {TASK_IDS}")
    return FAMILIES[name], int(level)


def generate(task_id: str, seed: int) -> TaskInstance:
    family, level = parse_task_id(task_id)
    return family.generate(level, random.Random(f"{task_id}:{seed}"))


__all__ = ["AnswerField", "FAMILIES", "TASK_IDS", "TaskInstance", "generate", "grade", "parse_task_id"]
