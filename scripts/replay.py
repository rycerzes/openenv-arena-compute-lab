# SPDX-License-Identifier: BSD-3-Clause

"""Replay every task against a running server over WebSocket.

Checks, per task: the reference solver (run through the python tool) scores 1.0,
the submission's example actions end the episode with a reward in [0, 1], and a
wrong answer scores below 1.

Usage: python scripts/replay.py --url http://localhost:8000 [--seeds 3]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

from solutions import solver_code  # noqa: E402

from compute_lab import ComputeLabAction, ComputeLabEnv  # noqa: E402
from compute_lab.server.tasks import TASK_IDS, parse_task_id  # noqa: E402

EXAMPLE_ACTIONS = [{"tool": "submit", "answer": {}}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tasks", default=",".join(TASK_IDS))
    args = ap.parse_args()
    failures = 0
    with ComputeLabEnv(base_url=args.url, message_timeout_s=120).sync() as env:
        env.reset(task_id="graph_paths-l1", seed=0)
        res = env.step(ComputeLabAction(tool="python", code=(
            "import os\nprint('uid', os.getuid())\n"
            "try:\n    open('/opt/app/server/app.py').read(); print('source readable')\n"
            "except Exception as e:\n    print('source blocked', type(e).__name__)\n")))
        print("sandbox probe:", res.observation.output.replace("\n", " | "))
        for task_id in args.tasks.split(","):
            family, _ = parse_task_id(task_id)
            for seed in range(args.seeds):
                obs = env.reset(task_id=task_id, seed=seed).observation
                out = env.step(ComputeLabAction(tool="python", code=solver_code(family.name, obs.output)))
                try:
                    answer = json.loads(out.observation.output.strip().splitlines()[-1])
                except (json.JSONDecodeError, IndexError):
                    print("FAIL", task_id, seed, "solver output:", out.observation.output[-200:])
                    failures += 1
                    continue
                final = env.step(ComputeLabAction(tool="submit", answer=answer))
                wrong = {k: (v * 1.1 + 7 if isinstance(v, (int, float)) else "x") for k, v in answer.items()}
                env.reset(task_id=task_id, seed=seed)
                bad = env.step(ComputeLabAction(tool="submit", answer=wrong))
                env.reset(task_id=task_id)
                for a in EXAMPLE_ACTIONS:
                    ex = env.step(ComputeLabAction(**a))
                ok = final.done and final.reward == 1.0 and bad.done and bad.reward < 1 and ex.done and 0 <= ex.reward <= 1
                failures += not ok
                if not ok:
                    print("FAIL", task_id, seed, final.reward, bad.reward, ex.done, ex.reward)
            print(f"{task_id}: ok" if not failures else f"{task_id}: failures so far {failures}", flush=True)
    print("FAILURES", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
