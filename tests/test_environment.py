# SPDX-License-Identifier: BSD-3-Clause

import json
import random
import re

import pytest

from compute_lab.models import ComputeLabAction
from compute_lab.server.compute_lab_environment import MAX_TOOL_CALLS, ComputeLabEnvironment
from compute_lab.server.tasks import TASK_IDS as V1_TASK_IDS
from compute_lab.server.tasks import generate as v1_generate
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
from compute_lab.server.workbench import DIFFICULTIES, FAMILIES, TASK_IDS, make_episode

from solutions import SOLVERS, solver_code

SEEDS = [0, 1, 7]
PINNED = [f"{name}-d{d}" for name in FAMILIES for d in DIFFICULTIES]
V1_NAMES = set(SOLVERS)


@pytest.fixture
def env():
    e = ComputeLabEnvironment()
    yield e
    e.close()


def act(env, tool, **args):
    return env.step(ComputeLabAction(tool=tool, args=args))


def submit(env, answer):
    return act(env, "submit", answer=answer)


@pytest.mark.parametrize("task_id", PINNED)
@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_scores_one_and_empty_scores_less(env, task_id, seed):
    episode, _ = make_episode(task_id, seed)
    oracle = episode.oracle
    episode.close()
    env.reset(task_id=task_id, seed=seed)
    for a in oracle:
        obs = env.step(ComputeLabAction(**a))
    assert obs.done and obs.reward == 1.0, obs.output
    env.reset(task_id=task_id, seed=seed)
    obs = submit(env, {})
    assert obs.done and obs.reward < 1.0


@pytest.mark.parametrize("task_id", [t for t in PINNED if t.rpartition("-d")[0] in V1_NAMES and not t.endswith("-d4")])
@pytest.mark.parametrize("seed", SEEDS)
def test_v1_reference_solvers_through_python_tool(env, task_id, seed):
    obs = env.reset(task_id=task_id, seed=seed)
    out = act(env, "python", code=solver_code(task_id.rpartition("-d")[0], obs.output))
    answer = json.loads(out.output.strip().splitlines()[-1])
    final = submit(env, answer)
    assert final.done and final.reward == 1.0, (final.output, answer)


SCHEDULE_SOLVER = r"""
import csv, itertools, json, os, re
jobs = {r["job"]: (int(r["release_min"]), int(r["processing_min"]), int(r["due_min"]), int(r["weight"])) for r in csv.DictReader(open("jobs.csv"))}
setups = {(r["from"], r["to"]): int(r["setup_min"]) for r in csv.DictReader(open("setups.csv"))} if os.path.exists("setups.csv") else {}
memo = open("memo.txt").read() if os.path.exists("memo.txt") else ""
w = re.search(r"down from t = (\d+) to t = (\d+)", memo)
window = (int(w.group(1)), int(w.group(2))) if w else None
for j in re.findall(r"(J\d+) was cancelled", memo):
    jobs.pop(j)
def cost(order):
    t, prev, c = 0, "IDLE", 0
    for j in order:
        r, p, d, wt = jobs[j]
        s = setups.get((prev, j), 0)
        start = max(t, r)
        if window and start < window[1] and start + s + p > window[0]:
            start = window[1]
        t = start + s + p
        c += wt * max(0, t - d)
        prev = j
    return c
print(json.dumps({"order": list(min(itertools.permutations(jobs), key=cost))}))
"""

SUBTITLE_SOLVER = r"""
import json, re
def ms(t):
    h, m, s, x = map(int, re.match(r"(\d+):(\d+):(\d+),(\d+)", t).groups())
    return ((h * 60 + m) * 60 + s) * 1000 + x
def ts(v):
    h, r = divmod(v, 3600000); m, r = divmod(r, 60000); s, x = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{x:03d}"
cues = {}
for block in open("movie.srt").read().strip().split("\n\n"):
    _, times, text = block.split("\n", 2)
    a, b = times.split(" --> ")
    cues[text] = (ms(a), ms(b))
notes = open("notes.md").read()
syncs = [(t, ms(v)) for t, v in re.findall(r'cue "(.+?)" should start at (\S+)\.', notes)]
fps = re.search(r"timed for ([\d.]+) fps; the release runs at ([\d.]+) fps", notes)
scene = re.search(r"adds a (\d+) ms scene; it affects cues that start after original time (\S+)\.", notes)
cut, extra = (ms(scene.group(2)), int(scene.group(1))) if scene else (None, 0)
if fps:
    k = float(fps.group(1)) / float(fps.group(2))
    t, target = syncs[0]
    off = target - cues[t][0] * k - (extra if cut is not None and cues[t][0] > cut else 0)
    f = lambda x, start: x * k + off + (extra if cut is not None and start > cut else 0)
else:
    (t1, s1), (t2, s2) = syncs
    a = (s2 - s1) / (cues[t2][0] - cues[t1][0]); b = s1 - a * cues[t1][0]
    f = lambda x, start: a * x + b
ans = {}
for key, which, text in re.findall(r'^- ((start|end)_\d+): \w+ of the cue "(.+?)"', PROMPT, re.M):
    s, e = cues[text]
    ans[key] = ts(int(f(s if which == "start" else e, s) + 0.5))
print(json.dumps(ans))
"""


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("family,code", [("machine_schedule", SCHEDULE_SOLVER), ("subtitle_sync", SUBTITLE_SOLVER)])
def test_python_solvers_for_new_families(env, family, code, difficulty, seed):
    obs = env.reset(task_id=f"{family}-d{difficulty}", seed=seed)
    out = act(env, "python", code=f"PROMPT = {obs.output!r}\n{code}")
    final = submit(env, json.loads(out.output.strip().splitlines()[-1]))
    assert final.reward == 1.0, final.output


def _local(day_text, hhmm):
    import datetime as dt

    return dt.datetime.strptime(f"{day_text} {hhmm}", "%a %d %b %Y %H:%M")


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_inbox_solvable_from_emails(env, difficulty, seed):
    import datetime as dt

    env.reset(task_id=f"inbox_scheduling-d{difficulty}", seed=seed)
    brief = act(env, "read", path="brief.txt").output
    duration = int(re.search(r"(\d+)-minute", brief).group(1))
    people = re.search(r"meeting with (.+?) between", brief).group(1).split(", ")
    ids = [int(x) for x in re.findall(r"^(\d+):", act(env, "list_emails").output, re.M)]
    offsets, busy, morning, room = {}, {p: [] for p in people}, set(), None
    corrections = []
    for i in ids:
        mail = act(env, "read_email", id=i).output
        sender = re.search(r"From: (\S+)", mail).group(1)
        if m := re.search(r"my local time \(UTC([+-]\d\d):00\)", mail):
            offsets[sender] = int(m.group(1))
            for d, a, b in re.findall(r"- (\w{3} \d\d \w{3} \d{4}) (\d\d:\d\d) to (\d\d:\d\d)", mail):
                busy[sender].append((_local(d, a), _local(d, b)))
        if "end by 12:00" in mail:
            morning.add(sender)
        if m := re.search(r"my (\w{3} \d\d \w{3} \d{4} \d\d:\d\d) block moved to (\w{3} \d\d \w{3} \d{4} \d\d:\d\d)", mail):
            corrections.append((sender, m.group(1), m.group(2)))
        if sender == "Facilities":
            room = [(_local(d, a), _local(d, b)) for d, a, b in re.findall(r"- (\w{3} \d\d \w{3} \d{4}) (\d\d:\d\d) to (\d\d:\d\d)", mail)]
    for p, old, new in corrections:
        o = dt.datetime.strptime(old, "%a %d %b %Y %H:%M")
        n = dt.datetime.strptime(new, "%a %d %b %Y %H:%M")
        busy[p] = [(n, n + (e - s)) if s == o else (s, e) for s, e in busy[p]]
    first_day = re.search(r"between (\w{3} \d\d \w{3} \d{4})", brief).group(1)
    start = dt.datetime.strptime(first_day, "%a %d %b %Y")
    for step in range(3 * 96):
        t = start + dt.timedelta(minutes=15 * step)  # organizer time, UTC+1
        end = t + dt.timedelta(minutes=duration)
        good = True
        for p in people:
            shift = dt.timedelta(hours=offsets[p] - 1)
            ls, le = t + shift, end + shift
            day = ls.replace(hour=0, minute=0)
            if ls < day + dt.timedelta(hours=9) or le > day + dt.timedelta(hours=17):
                good = False
            if p in morning and le > day + dt.timedelta(hours=12):
                good = False
            if any(s < le and ls < e for s, e in busy[p]):
                good = False
        if room is not None and not any(s <= t and end <= e for s, e in room):
            good = False
        if good:
            break
    final = submit(env, {"start": t.strftime("%Y-%m-%d %H:%M")})
    assert final.reward == 1.0, final.output


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_forensics_solvable_with_grep(env, difficulty, seed):
    obs = env.reset(task_id=f"auth_forensics-d{difficulty}", seed=seed)
    subnet = "/24" in obs.output
    files = sorted(re.findall(r"^(\S+) \(\d+ lines\)", act(env, "list").output, re.M), key=lambda f: f != "auth.log.1")
    sources = set()
    for name in files:
        while True:  # grep shows 30 lines: exclude addresses already seen until nothing new matches
            skip = "|".join(re.escape(ip) for ip in sources)
            pattern = r"Failed password for invalid user \S+ from " + (f"(?!(?:{skip}) )" if skip else "")
            found = set(re.findall(r"from (\S+) port", act(env, "grep", pattern=pattern, path=name).output))
            if not found - sources:
                break
            sources |= found
    key = (lambda ip: ip.rsplit(".", 1)[0]) if subnet else (lambda ip: ip)
    bad = {key(ip) for ip in sources}
    for name in files:
        out = act(env, "grep", pattern=r"Accepted password", path=name).output
        hits = [m for m in re.finditer(r"^\d+: (\w{3} \d\d \d\d:\d\d:\d\d) \S+ sshd\[\d+\]: Accepted password for (\S+) from (\S+)", out, re.M) if key(m.group(3)) in bad]
        if hits:
            m = hits[0]
            break
    final = submit(env, {"attacker_ip": m.group(3), "user": m.group(2), "time": m.group(1)})
    assert final.reward == 1.0, final.output


@pytest.mark.parametrize("difficulty", [1, 2, 3])
@pytest.mark.parametrize("seed", SEEDS)
def test_code_repair_bugs_visible_and_unfixed_scores_zero(env, difficulty, seed):
    env.reset(task_id=f"code_repair-d{difficulty}", seed=seed)
    report = act(env, "run_tests").output
    passed, total = map(int, re.search(r"^(\d+)/(\d+) visible tests passed", report).groups())
    assert passed < total, report
    assert submit(env, {}).reward == 0.0


def test_code_repair_breaking_a_correct_function_scores_zero(env):
    episode, _ = make_episode("code_repair-d3", 4)
    fixed = episode.oracle[0]["args"]["content"]
    episode.close()
    env.reset(task_id="code_repair-d3", seed=4)
    act(env, "write", path="lib.py", content=fixed.replace("return ", "return None and "))
    assert submit(env, {}).reward == 0.0


def test_code_repair_edit_requires_unique_match(env):
    env.reset(task_id="code_repair-d2", seed=0)
    out = act(env, "edit", path="lib.py", old="def ", new="def  ").output
    assert out.startswith("Error:") and "matches" in out


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_every_task_id_resets_and_is_deterministic(task_id):
    a, da = make_episode(task_id, 3)
    b, db = make_episode(task_id, 3)
    assert da == db and a.question == b.question and a.oracle == b.oracle
    a.close()
    b.close()


def test_random_difficulty_varies_across_seeds():
    assert {make_episode("subtitle_sync", s)[1] for s in range(40)} == set(DIFFICULTIES)


def test_tool_errors_and_unknown_tools_do_not_end_episode(env):
    env.reset(task_id="auth_forensics-d1", seed=0)
    for obs in [act(env, "read"), act(env, "grep", pattern="(", path="auth.log"), act(env, "submit")]:
        assert not obs.done and obs.reward == 0.0
    assert "Unknown tool" in act(env, "python", code="1").output


def test_budget_runs_out_with_zero_reward(env):
    env.reset(task_id="graph_paths-d1", seed=1)
    for _ in range(MAX_TOOL_CALLS):
        obs = act(env, "list")
    assert obs.done and obs.reward == 0.0 and obs.steps_left == 0


def test_python_timeout_and_clipping(env):
    env.reset(task_id="graph_paths-d1", seed=1)
    assert "killed" in act(env, "python", code="while True: pass").output
    assert len(act(env, "python", code="print('x' * 100000)").output) < 2000


def test_unknown_task_rejected(env):
    with pytest.raises(ValueError):
        env.reset(task_id="nope")
    with pytest.raises(ValueError):
        env.reset(task_id="code_repair-d9")


# v1 generator invariants (the v1 families are still served)

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
    rng = random.Random(6)
    for _ in range(20):
        lo, m, q, digit = rng.randrange(1, 3000), rng.randrange(3, 13), rng.randrange(4, 10), rng.randrange(1, 10)
        hi = lo + rng.randrange(1, 4000)
        r, s = rng.randrange(m), rng.randrange(q)
        dp = count_digit_dp(hi, m, r, q, s, digit) - count_digit_dp(lo - 1, m, r, q, s, digit)
        brute = sum(1 for n in range(lo, hi + 1) if n % m == r and digit_sum(n) % q == s and str(digit) not in str(n))
        assert dp == brute
        assert count_digit_dp(hi, m, r, q, s) - count_digit_dp(lo - 1, m, r, q, s) == count_brute(lo, hi, m, r, q, s)


def test_fast_recurrences_match_loops():
    rng = random.Random(9)
    for _ in range(20):
        mod = rng.randrange(10, 10**9)
        n = rng.randrange(0, 300)
        args = [rng.randrange(mod), rng.randrange(mod)] + [rng.randrange(1, 1000) for _ in range(4)]
        assert recurrence_fast(*args, mod, n) == recurrence_loop(*args, mod, n)
        cargs = [rng.randrange(mod) for _ in range(2)] + [rng.randrange(1, 1000) for _ in range(6)]
        assert coupled_fast(*cargs, mod, n) == coupled_loop(*cargs, mod, n)


@pytest.mark.parametrize("task_id", [t for t in V1_TASK_IDS if t != "constrained_count-l2"])
def test_v1_generation_never_crashes(task_id):
    for seed in range(150):
        v1_generate(task_id, seed)


def test_python_tool_is_safe_under_concurrent_threads():
    # preexec_fn-based sandboxes could deadlock between fork and exec when threads run tools at once.
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        e = ComputeLabEnvironment()
        try:
            e.reset(task_id="graph_paths-d1", seed=i)
            return act(e, "python", code="print(sum(range(10)))").output.strip()
        finally:
            e.close()

    with ThreadPoolExecutor(16) as pool:
        outs = list(pool.map(one, range(32)))
    assert outs == ["45"] * 32


def test_python_tool_errors_report_traceback(env):
    env.reset(task_id="graph_paths-d1", seed=0)
    out = act(env, "python", code="1/0").output
    assert "ZeroDivisionError" in out


@pytest.mark.parametrize("task_id", ["reaction_kinetics-d1", "stoichiometry-d1", "fx_ledger-d4", "xnpv_xirr-d4"])
def test_guessing_a_categorical_field_earns_nothing(env, task_id):
    # Partial credit let a policy score by guessing the reaction order / limiting reagent / top category.
    episode, _ = make_episode(task_id, 5)
    answer = dict(episode.oracle[0]["args"]["answer"])
    episode.close()
    numeric = next(k for k, v in answer.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "order")
    answer[numeric] = -12345.0
    env.reset(task_id=task_id, seed=5)
    assert submit(env, answer).reward == 0.0


def test_read_returns_whole_lines_and_a_resume_point(env):
    # Clipping the middle of a long read once hid the line an agent needed.
    env.reset(task_id="auth_forensics-d3", seed=1)
    out = act(env, "read", path="auth.log").output
    body, marker = out.rsplit("\n", 1)
    assert "chars cut" not in out and marker.startswith("[") and "pass start=" in marker
    nxt = int(re.search(r"start=(\d+)", marker).group(1))
    assert all(re.match(r"\w{3} \d\d \d\d:\d\d:\d\d ", line) for line in body.splitlines())
    follow = act(env, "read", path="auth.log", start=nxt).output
    assert follow.splitlines()[0] != body.splitlines()[0]


def test_unseeded_resets_share_an_instance_within_a_window(env, monkeypatch):
    import compute_lab.server.compute_lab_environment as cle

    seed_of = lambda task_id, **kw: (env.reset(task_id=task_id, **kw), env.state.seed)[1]
    monkeypatch.setattr(cle.time, "time", lambda: 1_800_000_000.0)
    first = seed_of("inbox_scheduling")
    assert seed_of("inbox_scheduling") == first
    assert seed_of("subtitle_sync") != first
    monkeypatch.setattr(cle.time, "time", lambda: 1_800_000_000.0 + cle.GROUP_WINDOW_S)
    assert seed_of("inbox_scheduling") != first
    assert seed_of("inbox_scheduling", seed=5) == 5
