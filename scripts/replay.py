# SPDX-License-Identifier: BSD-3-Clause

"""Replay tasks against a running server over WebSocket.

For each task and seed, the episode is also generated locally (generation is
deterministic), and its oracle actions are played against the server: they must
score 1.0. An empty submission must score below 1, and the submission's example
action must end the episode with a reward in [0, 1]. A probe checks that the
python tool runs as nobody without access to the environment source.

Usage: python scripts/replay.py --url http://localhost:8000 [--seeds 3] [--tasks a-d1,b-d2]
"""

import argparse
import sys

from compute_lab import ComputeLabAction, ComputeLabEnv
from compute_lab.server.workbench import DIFFICULTIES, FAMILIES, make_episode

EXAMPLE_ACTIONS = [{"tool": "submit", "args": {"answer": {}}}]
PROBE = (
    "import os\nprint('uid', os.getuid())\n"
    "try:\n    open('/opt/app/server/app.py').read(); print('source readable')\n"
    "except Exception as e:\n    print('source blocked', type(e).__name__)\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tasks", default=",".join(f"{f}-d{d}" for f in FAMILIES for d in DIFFICULTIES))
    args = ap.parse_args()
    failures = 0
    with ComputeLabEnv(base_url=args.url, message_timeout_s=120).sync() as env:
        env.reset(task_id="graph_paths-d1", seed=0)
        probe = env.step(ComputeLabAction(tool="python", args={"code": PROBE}))
        print("sandbox probe:", probe.observation.output.replace("\n", " | "))
        for task_id in args.tasks.split(","):
            for seed in range(args.seeds):
                episode, _ = make_episode(task_id, seed)
                oracle = episode.oracle
                episode.close()
                env.reset(task_id=task_id, seed=seed)
                for action in oracle:
                    final = env.step(ComputeLabAction(**action))
                env.reset(task_id=task_id, seed=seed)
                empty = env.step(ComputeLabAction(tool="submit", args={"answer": {}}))
                env.reset(task_id=task_id)
                for action in EXAMPLE_ACTIONS:
                    example = env.step(ComputeLabAction(**action))
                ok = (
                    final.done and final.reward == 1.0
                    and empty.done and empty.reward < 1.0
                    and example.done and 0.0 <= example.reward <= 1.0
                )
                if not ok:
                    failures += 1
                    print("FAIL", task_id, seed, final.reward, final.observation.output[:120], empty.reward, example.done)
            print(f"{task_id}: {'ok' if not failures else f'{failures} failures so far'}", flush=True)
    print("FAILURES", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
