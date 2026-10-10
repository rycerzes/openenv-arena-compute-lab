# SPDX-License-Identifier: BSD-3-Clause

import json
import random

import pytest

from compute_lab.models import ComputeLabAction
from compute_lab.server.compute_lab_environment import MAX_TOOL_CALLS, ComputeLabEnvironment
from compute_lab.server.tasks import TASK_IDS, generate, grade, parse_task_id
from compute_lab.server.tasks.math_tasks import (
    count_brute,
    count_digit_dp,
    coupled_fast,
    coupled_loop,
    digit_sum,
    recurrence_fast,
    recurrence_loop,
)
from compute_lab.server.tasks.science import REACTIONS, parse_formula

from solutions import solver_code

SEEDS = [0, 1, 7, 42]


@pytest.fixture
def env():
    e = ComputeLabEnvironment()
    yield e
    e.close()


def submit(env, answer):
    return env.step(ComputeLabAction(tool="submit", answer=answer))


@pytest.mark.parametrize("task_id", TASK_IDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_reference_solver_through_python_tool_scores_one(env, task_id, seed):
    obs = env.reset(task_id=task_id, seed=seed)
    family, _ = parse_task_id(task_id)
    out = env.step(ComputeLabAction(tool="python", code=solver_code(family.name, obs.output)))
    assert not out.done, out.output
    answer = json.loads(out.output.strip().splitlines()[-1])
    final = submit(env, answer)
    assert final.done and final.reward == 1.0, (final.output, answer)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_wrong_and_empty_answers_score_below_one(task_id):
    task = generate(task_id, 3)
    assert grade(task.fields, {})[0] == 0.0
    wrong = {}
    for f in task.fields:
        if f.kind == "text":
            wrong[f.name] = next((c for c in f.choices if c != f.expected), "nope")
        else:
            wrong[f.name] = float(f.expected) * 1.07 + 3
    assert grade(task.fields, wrong)[0] == 0.0


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_same_seed_same_instance_and_seeds_differ(task_id):
    a, b, c = generate(task_id, 11), generate(task_id, 11), generate(task_id, 12)
    assert a.files == b.files and [f.expected for f in a.fields] == [f.expected for f in b.fields]
    assert a.files != c.files


def test_reactions_are_balanced():
    for reactants, products in REACTIONS:
        totals = []
        for side in (reactants, products):
            count = {}
            for coef, formula in side:
                for el, n in parse_formula(formula).items():
                    count[el] = count.get(el, 0) + coef * n
            totals.append(count)
        assert totals[0] == totals[1], (reactants, products)


def test_digit_dp_matches_brute_force():
    rng = random.Random(5)
    for _ in range(20):
        hi, m, q = rng.randrange(1, 5000), rng.randrange(3, 13), rng.randrange(4, 10)
        r, s = rng.randrange(m), rng.randrange(q)
        dp = count_digit_dp(hi, m, r, q, s) - count_digit_dp(0, m, r, q, s)
        assert dp == count_brute(1, hi, m, r, q, s)


def test_digit_dp_with_forbidden_digit_matches_brute_force():
    rng = random.Random(6)
    for _ in range(20):
        lo, m, q, digit = rng.randrange(1, 3000), rng.randrange(3, 13), rng.randrange(4, 10), rng.randrange(1, 10)
        hi = lo + rng.randrange(1, 4000)
        r, s = rng.randrange(m), rng.randrange(q)
        dp = count_digit_dp(hi, m, r, q, s, digit) - count_digit_dp(lo - 1, m, r, q, s, digit)
        brute = sum(1 for n in range(lo, hi + 1) if n % m == r and digit_sum(n) % q == s and str(digit) not in str(n))
        assert dp == brute


def test_coupled_recurrence_matches_loop():
    rng = random.Random(10)
    for _ in range(20):
        mod = rng.randrange(10, 10**9)
        args = [rng.randrange(mod) for _ in range(2)] + [rng.randrange(1, 1000) for _ in range(6)]
        n = rng.randrange(0, 300)
        assert coupled_fast(*args, mod, n) == coupled_loop(*args, mod, n)


def test_fast_recurrence_matches_loop():
    rng = random.Random(9)
    for _ in range(20):
        mod = rng.randrange(10, 10**9)
        args = [rng.randrange(mod), rng.randrange(mod)] + [rng.randrange(1, 1000) for _ in range(4)]
        n = rng.randrange(0, 300)
        assert recurrence_fast(*args, mod, n) == recurrence_loop(*args, mod, n)


def test_partial_credit(env):
    env.reset(task_id="fx_ledger-l3", seed=1)
    fields = env._task.fields
    obs = submit(env, {fields[0].name: fields[0].expected})
    assert obs.done and obs.reward == 0.5


def test_finish_action_ends_with_zero(env):
    env.reset(task_id="recurrence_mod-l1", seed=1)
    obs = submit(env, {})
    assert obs.done and obs.reward == 0.0


def test_example_actions_end_every_task(env):
    for task_id in TASK_IDS:
        env.reset(task_id=task_id, seed=2)
        obs = env.step(ComputeLabAction(tool="read", path="rules.md"))
        assert not obs.done
        obs = submit(env, {})
        assert obs.done and 0.0 <= obs.reward <= 1.0


def test_budget_runs_out_with_zero_reward(env):
    env.reset(task_id="graph_paths-l1", seed=1)
    for i in range(MAX_TOOL_CALLS):
        obs = env.step(ComputeLabAction(tool="read", path="edges.csv"))
    assert obs.done and obs.reward == 0.0 and obs.steps_left == 0
    assert env.step(ComputeLabAction(tool="read", path="edges.csv")).done


def test_tool_errors_do_not_end_episode(env):
    env.reset(task_id="graph_paths-l1", seed=1)
    for action in [
        ComputeLabAction(tool="calc"),
        ComputeLabAction(tool="read", path="../etc/passwd"),
        ComputeLabAction(tool="python"),
        ComputeLabAction(tool="submit"),
    ]:
        obs = env.step(action)
        assert not obs.done and obs.reward == 0.0


def test_python_timeout_and_clipping(env):
    env.reset(task_id="graph_paths-l1", seed=1)
    obs = env.step(ComputeLabAction(tool="python", code="while True: pass"))
    assert "killed" in obs.output
    obs = env.step(ComputeLabAction(tool="python", code="print('x' * 100000)"))
    assert len(obs.output) < 2000


def test_long_files_are_truncated_on_read(env):
    env.reset(task_id="graph_paths-l3", seed=1)
    obs = env.step(ComputeLabAction(tool="read", path="edges.csv"))
    assert "more lines; process the file with python" in obs.output


def test_unknown_task_rejected(env):
    with pytest.raises(ValueError):
        env.reset(task_id="nope-l1")


@pytest.mark.parametrize("seed", [53, 173])
def test_zero_order_run_b_never_goes_non_positive(seed):
    # These seeds used to drive a zero-order run B below zero and crash the fit.
    task = generate("reaction_kinetics-l3", seed)
    assert all(f.expected is not None for f in task.fields)


@pytest.mark.parametrize("task_id", [t for t in TASK_IDS if t != "constrained_count-l2"])
def test_generation_never_crashes(task_id):
    for seed in range(200):
        generate(task_id, seed)
