# SPDX-License-Identifier: BSD-3-Clause

"""Build the public Hugging Face dataset folder for the environment.

Writes dataset/: tasks.jsonl (one row per pinned task id, with a seed-0 example
prompt and oracle), a source snapshot, the calibration summary and the card.

Usage: python scripts/make_dataset.py --tasks a-d1,b-d2 --calibration calibration/summary_v2.json
"""

import argparse
import json
import os
import shutil

from compute_lab.server.compute_lab_environment import render_prompt
from compute_lab.server.workbench import DIFFICULTIES, FAMILIES, make_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_FILES = ["__init__.py", "models.py", "client.py", "openenv.yaml", "pyproject.toml", "requirements.lock.txt", "server", "scripts", "tests"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="task ids in the submission (split=train)")
    ap.add_argument("--calibration", default=os.path.join(ROOT, "calibration", "summary_v3.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "dataset"))
    args = ap.parse_args()
    submitted = set(args.tasks.split(","))
    calibration = json.load(open(args.calibration)).get("tasks", {}) if os.path.exists(args.calibration) else {}

    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out)
    with open(os.path.join(args.out, "tasks.jsonl"), "w") as f:
        for name, family in FAMILIES.items():
            for difficulty in DIFFICULTIES:
                task_id = f"{name}-d{difficulty}"
                episode, _ = make_episode(task_id, 0)
                row = {
                    "task_id": task_id,
                    "family": name,
                    "difficulty": difficulty,
                    "domain": family.domain,
                    "summary": family.summary,
                    "tools": [t.name for t in episode.tools] + ["submit"],
                    "in_submission": task_id in submitted,
                    "split": "train" if task_id in submitted else "unused",
                    "calibration": calibration.get(task_id),
                    "example_seed": 0,
                    "example_prompt": render_prompt(episode),
                    "example_oracle": json.dumps(episode.oracle),
                }
                episode.close()
                f.write(json.dumps(row) + "\n")
    for name in SOURCE_FILES:
        src, dst = os.path.join(ROOT, name), os.path.join(args.out, "source", name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "raw", "*.pyc"))
        elif os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
    if os.path.exists(args.calibration):
        shutil.copy(args.calibration, os.path.join(args.out, "calibration_summary.json"))
    shutil.copy(os.path.join(ROOT, "README.md"), os.path.join(args.out, "README.md"))
    if os.path.exists(os.path.join(ROOT, "calibration", "format_ab.json")):
        shutil.copy(os.path.join(ROOT, "calibration", "format_ab.json"), os.path.join(args.out, "format_ab.json"))
    print(f"wrote {args.out}: {len(FAMILIES) * len(DIFFICULTIES)} pinned tasks ({len(submitted)} in submission)")


if __name__ == "__main__":
    main()
