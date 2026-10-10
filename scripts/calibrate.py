# SPDX-License-Identifier: BSD-3-Clause

"""Play Qwen3.8-27B (thinking off) against the environment the way the Arena trainer does.

The model gets a system message with the action schema, the reset observation as
the first user turn and each later observation as the next user turn, all as
JSON. Each reply must be one JSON object: an action or {"finish": true}. Any
other reply ends the episode, which is then scored through the finish action.

Uses the Hugging Face login already saved on this machine (Inference Providers).

Usage:
    python scripts/calibrate.py --episodes 4 --out calibration/run1.jsonl [--tasks a-l1,b-l2]
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import InferenceClient
from pydantic import ValidationError

from compute_lab.models import ComputeLabAction
from compute_lab.server.compute_lab_environment import ComputeLabEnvironment
from compute_lab.server.tasks import TASK_IDS

MODEL = "Qwen/Qwen3.8-27B"
FINISH_ACTION = {"tool": "submit", "answer": {}}
ACTION_SCHEMA = ComputeLabAction.model_json_schema()
SYSTEM = (
    "You are an agent acting in an environment. Each user message is a JSON observation. "
    "Reply with exactly one JSON object and nothing else: either an action that matches this "
    "JSON schema, or {\"finish\": true} to end the episode.\n"
    f"Action schema: {json.dumps(ACTION_SCHEMA)}"
)


def parse_reply(text: str):
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return None, "no_json"
    try:
        obj, end = json.JSONDecoder(strict=False).raw_decode(text[start:])
    except json.JSONDecodeError:
        return None, "bad_json"
    if not isinstance(obj, dict):
        return None, "not_object"
    clean = start == 0 and not text[start + end :].strip()
    return obj, "ok" if clean else "extra_text"


def play(client, task_id, seed, max_tokens, completion_budget, context_budget):
    env = ComputeLabEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": obs.model_dump_json(exclude={"metadata"})},
    ]
    log = {"task_id": task_id, "seed": seed, "turns": [], "end": None}
    used = 0
    while True:
        try:
            r = client.chat_completion(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=1.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as e:  # provider error: drop the episode, don't score it
            log["end"] = f"provider_error:{type(e).__name__}"
            log["reward"] = None
            env.close()
            return log
        reply = r.choices[0].message.content or ""
        used += r.usage.completion_tokens
        action, fmt = parse_reply(reply)
        log["turns"].append({"reply": reply[:1500], "format": fmt, "finish_reason": r.choices[0].finish_reason})
        if action == {"finish": True} or action is None:
            log["end"] = "finish" if action else fmt
            action = FINISH_ACTION
        elif set(action) == {"action"} and isinstance(action["action"], dict):
            action = action["action"]
        try:
            act = ComputeLabAction(**action)
        except (ValidationError, TypeError):
            log["end"] = "schema_invalid"
            act = ComputeLabAction(**FINISH_ACTION)
        obs = env.step(act)
        obs_json = obs.model_dump_json(exclude={"metadata"})
        log["turns"][-1]["obs"] = obs.output[:600]
        if obs.done:
            if not log["end"]:
                log["end"] = "submitted" if act.tool == "submit" else "out_of_calls"
            break
        if log["end"]:
            obs = env.step(ComputeLabAction(**FINISH_ACTION))
            break
        used += len(obs_json) // 3
        if used > completion_budget or r.usage.prompt_tokens + len(obs_json) // 3 > context_budget:
            log["end"] = "token_budget"
            obs = env.step(ComputeLabAction(**FINISH_ACTION))
            break
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": obs_json}]
    log["reward"] = obs.reward
    log["steps"] = len(log["turns"])
    env.close()
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--tasks", default=",".join(TASK_IDS))
    ap.add_argument("--provider", default="deepinfra")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--completion-budget", type=int, default=8192)
    ap.add_argument("--context-budget", type=int, default=16384)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    client = InferenceClient(provider=args.provider, timeout=180)
    rng = random.SystemRandom()
    jobs = [(t, rng.randrange(2**31)) for t in args.tasks.split(",") for _ in range(args.episodes)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    with open(args.out, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(play, client, t, s, args.max_tokens, args.completion_budget, args.context_budget) for t, s in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            log = fut.result()
            f.write(json.dumps(log) + "\n")
            f.flush()
            print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {log['task_id']} reward={log['reward']} steps={log.get('steps')} end={log['end']}", flush=True)


if __name__ == "__main__":
    main()
