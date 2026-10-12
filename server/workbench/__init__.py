# SPDX-License-Identifier: BSD-3-Clause

"""Workbench v2 task registry.

Task ids: `<family>` draws a difficulty from 1-4 on every reset (so training
groups stay mixed as the policy improves); `<family>-d<k>` pins difficulty k.
"""

import random

from . import auth_forensics, code_repair, expense_audit, inbox_scheduling, inventory_reorder, machine_schedule, subtitle_sync, v1_families
from .base import Episode, Family

FAMILIES: dict[str, Family] = {
    f.name: f
    for f in [
        *v1_families.FAMILIES,
        code_repair.FAMILY,
        machine_schedule.FAMILY,
        inbox_scheduling.FAMILY,
        auth_forensics.FAMILY,
        subtitle_sync.FAMILY,
        expense_audit.FAMILY,
        inventory_reorder.FAMILY,
    ]
}
DIFFICULTIES = (1, 2, 3, 4)
TASK_IDS = [*FAMILIES, *(f"{name}-d{d}" for name in FAMILIES for d in DIFFICULTIES)]


def parse_task_id(task_id: str) -> tuple[Family, int | None]:
    name, sep, level = task_id.rpartition("-d")
    if sep and name in FAMILIES and level in {str(d) for d in DIFFICULTIES}:
        return FAMILIES[name], int(level)
    if task_id in FAMILIES:
        return FAMILIES[task_id], None
    raise ValueError(f"unknown task_id {task_id!r}; expected a family ({', '.join(FAMILIES)}) optionally suffixed -d1..-d4")


def make_episode(task_id: str, seed: int) -> tuple[Episode, int]:
    family, difficulty = parse_task_id(task_id)
    rng = random.Random(f"{task_id}:{seed}")
    if difficulty is None:
        difficulty = rng.choice(DIFFICULTIES)
    return family.make(difficulty, rng), difficulty


__all__ = ["DIFFICULTIES", "FAMILIES", "TASK_IDS", "Episode", "Family", "make_episode", "parse_task_id"]
