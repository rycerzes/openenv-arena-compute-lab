# SPDX-License-Identifier: BSD-3-Clause

"""Summarize calibration logs: per-task mean reward, spread and episode endings."""

import json
import sys
from collections import Counter, defaultdict


def main(paths):
    by_task, ends, fmts = defaultdict(list), Counter(), Counter()
    for path in paths:
        for line in open(path):
            d = json.loads(line)
            if d["reward"] is None:
                ends["provider_error"] += 1
                continue
            by_task[d["task_id"]].append(d["reward"])
            ends[d["end"]] += 1
            fmts.update(t["format"] for t in d["turns"])
    print(f"{'task':28} {'n':>3} {'mean':>5} {'min':>4} {'max':>4}  rewards")
    for task in sorted(by_task):
        r = by_task[task]
        print(f"{task:28} {len(r):3d} {sum(r) / len(r):5.2f} {min(r):4.2f} {max(r):4.2f}  {[round(x, 2) for x in r]}")
    print("episode endings:", dict(ends))
    print("reply formats:", dict(fmts))


if __name__ == "__main__":
    main(sys.argv[1:])
