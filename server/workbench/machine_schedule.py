# SPDX-License-Identifier: BSD-3-Clause

"""Industrial and physical systems: sequence jobs on one machine.

Minimize total weighted tardiness. Higher difficulty adds sequence-dependent
setup times, a maintenance window and a cancelled job (both in memo.txt). Any
optimal order earns 1.0; anything else 0 (a near-miss tier would reward random orders).
"""

import itertools

from ..sandbox import Workspace
from ..tasks.base import csv_text
from .base import Episode, Family, file_tools, submit_action, submit_hint

RULES = """# Scheduling rules
- One machine processes one job at a time, without interruption, in the order you submit.
- A job cannot start before its release time. Times are minutes from shift start (t = 0).
- Before each job the machine needs the setup time from setups.csv for (previous job -> this job);
  the first job uses the row for IDLE. Without setups.csv there is no setup time.
- Setup and processing form one block. If a block would overlap a maintenance window, the
  block starts when the window ends.
- Tardiness of a job = max(0, completion time - due time). Cost = sum of weight * tardiness.
- Submit every active job exactly once. memo.txt, when present, overrides the other files.
"""


def schedule_cost(order, jobs, setups, window):
    t, prev, cost = 0, "IDLE", 0
    for j in order:
        release, proc, due, weight = jobs[j]
        setup = setups.get((prev, j), 0) if setups else 0
        start = max(t, release)
        if window and start < window[1] and start + setup + proc > window[0]:
            start = window[1]
        t = start + setup + proc
        cost += weight * max(0, t - due)
        prev = j
    return cost


def make(difficulty, rng):
    n = {1: 4, 2: 5, 3: 6, 4: 7}[difficulty]
    names = [f"J{i}" for i in range(1, n + 2)]
    jobs = {}
    t = 0
    for name in names:
        proc = rng.randrange(10, 60, 5)
        release = rng.choice([0, 0, rng.randrange(0, max(10, t), 5)])
        due = release + proc + rng.randrange(0, 120, 5)
        jobs[name] = (release, proc, due, rng.randint(1, 5))
        t += proc
    cancelled = rng.choice(names) if difficulty == 4 else None
    if difficulty < 4:
        jobs.pop(names[-1])
    active = [j for j in jobs if j != cancelled]
    setups = None
    if difficulty >= 2:
        setups = {(a, b): rng.randrange(0, 25, 5) for a in ["IDLE", *jobs] for b in jobs if a != b}
    window = None
    notes = []
    if difficulty >= 3:
        start = rng.randrange(40, 140, 10)
        window = (start, start + rng.randrange(20, 50, 10))
        notes.append(f"- Planned maintenance: the machine is down from t = {window[0]} to t = {window[1]}.")
    if cancelled:
        notes.append(f"- {cancelled} was cancelled by the customer. Do not schedule it.")

    costs = {order: schedule_cost(order, jobs, setups, window) for order in itertools.permutations(active)}
    best = min(costs.values())
    files = {
        "rules.md": RULES,
        "jobs.csv": csv_text(["job", "release_min", "processing_min", "due_min", "weight"], [[j, *jobs[j]] for j in jobs]),
    }
    if setups:
        files["setups.csv"] = csv_text(["from", "to", "setup_min"], [[a, b, s] for (a, b), s in setups.items()])
    if notes:
        files["memo.txt"] = "# Shift memo\n" + "\n".join(notes) + "\n- Coffee machine on floor 2 is out of order.\n"
    ws = Workspace(files)

    def grade(answer):
        order = answer.get("order") if isinstance(answer, dict) else None
        if not isinstance(order, list) or sorted(map(str, order)) != sorted(active):
            return 0.0, f"order must list each active job once ({len(active)} jobs)"
        cost = costs[tuple(map(str, order))]
        if cost == best:
            return 1.0, f"optimal (cost {cost})"
        return 0.0, f"cost {cost}; not optimal"

    return Episode(
        question="Choose the job order that minimizes total weighted tardiness, following rules.md.",
        tools=file_tools(ws, files),
        answer_hint=submit_hint({"order": ["J2", "J1", "..."]}),
        grade=grade,
        workspace=ws,
        oracle=[submit_action({"order": list(min(costs, key=costs.get))})],
    )


FAMILY = Family("machine_schedule", "industrial_physical_systems", "Single-machine sequencing with setups, maintenance and cancellations", make)
