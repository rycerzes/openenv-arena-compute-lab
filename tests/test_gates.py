# SPDX-License-Identifier: BSD-3-Clause

"""Quality gates over many seeds: no answer leakage, and no single answer that solves many instances."""

import json
from collections import Counter

import pytest

from compute_lab.server.compute_lab_environment import render_prompt
from compute_lab.server.workbench import DIFFICULTIES, FAMILIES, make_episode

SEEDS = range(100)
SLOW = {"constrained_count-d2"}  # brute-force generation; fewer seeds keep the suite fast
PINNED = [f"{name}-d{d}" for name in FAMILIES for d in DIFFICULTIES if name != "code_repair"]


def _answer(episode):
    return episode.oracle[-1]["args"]["answer"]


def _leaks(value, prompt):
    # Values offered as listed choices ("one of A, B, C") are not leaks: grading is all-or-nothing,
    # so naming the right category earns nothing without the computed fields.
    choices = {c.strip() for line in prompt.splitlines() if "one of " in line for c in line.split("one of ", 1)[1].split(",")}
    if isinstance(value, str):
        return len(value) >= 4 and value in prompt and value not in choices
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        digits = f"{value:.6g}".replace(".", "").replace("-", "").lstrip("0")
        return len(digits) >= 3 and f"{value:.6g}" in prompt
    if isinstance(value, list):
        return False
    return False


@pytest.mark.parametrize("task_id", PINNED)
def test_answer_never_appears_in_the_prompt(task_id):
    for seed in SEEDS if task_id not in SLOW else range(15):
        episode, _ = make_episode(task_id, seed)
        prompt = render_prompt(episode)
        leaked = [k for k, v in _answer(episode).items() if _leaks(v, prompt)]
        episode.close()
        assert not leaked, (task_id, seed, leaked)


@pytest.mark.parametrize("task_id", PINNED)
def test_most_common_answer_solves_at_most_ten_percent(task_id):
    seeds = SEEDS if task_id not in SLOW else range(15)
    episodes = [make_episode(task_id, s)[0] for s in seeds]
    try:
        common = Counter(json.dumps(_answer(e), sort_keys=True) for e in episodes).most_common(1)[0][0]
        solved = sum(e.grade(json.loads(common))[0] == 1.0 for e in episodes)
    finally:
        for e in episodes:
            e.close()
    assert solved <= 0.1 * len(episodes), (task_id, common, solved)
