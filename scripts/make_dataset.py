# SPDX-License-Identifier: BSD-3-Clause

"""Build the public Hugging Face dataset folder for the environment.

Writes dataset/: tasks.jsonl (one task per line, with a seed-0 example instance),
a source snapshot, the calibration summary and the dataset card (README.md is
written separately and copied in).

Usage: python scripts/make_dataset.py --tasks a-l1,b-l2 --calibration calibration/summary.json
"""

import argparse
import json
import os
import shutil

from compute_lab.server.compute_lab_environment import render_prompt
from compute_lab.server.tasks import FAMILIES, TASK_IDS, generate, parse_task_id

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_FILES = ["__init__.py", "models.py", "client.py", "openenv.yaml", "pyproject.toml", "requirements.lock.txt", "server", "scripts", "tests"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASK_IDS), help="task ids in the submission (split=train)")
    ap.add_argument("--calibration", default=os.path.join(ROOT, "calibration", "summary.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "dataset"))
    args = ap.parse_args()
    submitted = args.tasks.split(",")
    calibration = json.load(open(args.calibration)) if os.path.exists(args.calibration) else {}

    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out)
    with open(os.path.join(args.out, "tasks.jsonl"), "w") as f:
        for task_id in TASK_IDS:
            family, level = parse_task_id(task_id)
            example = generate(task_id, 0)
            row = {
                "task_id": task_id,
                "family": family.name,
                "level": level,
                "domain": family.domain,
                "summary": family.summary,
                "in_submission": task_id in submitted,
                "split": "train" if task_id in submitted else "unused",
                "calibration": calibration.get("tasks", {}).get(task_id),
                "example_seed": 0,
                "example_prompt": render_prompt(example),
                "example_files": json.dumps(example.files),
                "example_answer": json.dumps({fld.name: fld.expected for fld in example.fields}),
            }
            f.write(json.dumps(row) + "\n")
    for name in SOURCE_FILES:
        src = os.path.join(ROOT, name)
        dst = os.path.join(args.out, "source", name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "raw", "*.pyc"))
        elif os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
    if calibration:
        shutil.copy(args.calibration, os.path.join(args.out, "calibration_summary.json"))
    shutil.copy(os.path.join(ROOT, "README.md"), os.path.join(args.out, "README.md"))
    print(f"wrote {args.out}: {len(TASK_IDS)} tasks ({len(submitted)} in submission), families: {', '.join(FAMILIES)}")


if __name__ == "__main__":
    main()
