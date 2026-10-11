# SPDX-License-Identifier: BSD-3-Clause

"""Arena-style multi-turn rollouts shared by the local training pilots.

Mirrors the Arena trainer's protocol: a system message with the action schema,
the reset observation as the first user turn, each later observation as the next
user turn (JSON), and one JSON object per reply. Any other reply ends the episode,
which is then scored through the finish action. Qwen3 chat format, thinking off.
"""

import os
import statistics
import sys

from pydantic import ValidationError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import FINISH_ACTION, SYSTEM, parse_reply  # noqa: E402

from compute_lab.models import ComputeLabAction  # noqa: E402
from compute_lab.server.compute_lab_environment import ComputeLabEnvironment  # noqa: E402
from compute_lab.server.workbench import DIFFICULTIES, FAMILIES  # noqa: E402

FORMAT_FAILURES = ("bad_json", "no_json", "not_object", "extra_text", "schema_invalid")
DEFAULT_HELD_OUT = "graph_paths,xnpv_xirr,auth_forensics,inbox_scheduling"


def add_family_args(ap):
    ap.add_argument("--held-out", type=lambda v: v.split(","), default=DEFAULT_HELD_OUT.split(","))
    ap.add_argument("--train-families", type=lambda v: v.split(","), default=None, help="defaults to every family not held out")


def check_family_args(ap, args):
    """Default `--train-families` to every family not held out; reject unknown names."""
    if args.train_families is None:
        args.train_families = [f for f in FAMILIES if f not in args.held_out]
    unknown = set(args.train_families + args.held_out) - set(FAMILIES)
    if unknown:
        ap.error(f"unknown families: {sorted(unknown)}")


def eval_splits(train_families, held_out):
    """Pinned-difficulty task ids for the trained and the held-out families."""
    pinned = lambda fams: [f"{f}-d{d}" for f in fams for d in DIFFICULTIES]
    return {"train_families": pinned(train_families), "held_out_families": pinned(held_out)}


def eval_row(model, split, eps, seconds):
    """Summary of one evaluation split; `by_task` is the mean reward per task id."""
    by_task = {}
    for ep in eps:
        by_task.setdefault(ep["task_id"], []).append(ep["reward"])
    return {
        "model": model, "split": split, "episodes": len(eps),
        "mean_reward": round(statistics.mean(ep["reward"] for ep in eps), 4),
        "solved": sum(ep["reward"] == 1.0 for ep in eps),
        "format_failures": sum(ep["end"] in FORMAT_FAILURES for ep in eps),
        "context_ends": sum(ep["end"] == "context" for ep in eps),
        "mean_turns": round(statistics.mean(ep["turns"] for ep in eps), 2),
        "by_task": {t: round(statistics.mean(v), 3) for t, v in sorted(by_task.items())},
        "seconds": round(seconds),
    }


def vllm_completions(outputs):
    """`(token_ids, logprobs)` pairs from vLLM request outputs, with placeholder log-probs."""
    return [(o.outputs[0].token_ids, [0.0] * len(o.outputs[0].token_ids)) for o in outputs]


class Renderer:
    """Qwen3 chat rendering with thinking off, extended turn by turn so tokens stay a strict prefix."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def first(self, obs_json):
        text = self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": obs_json}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tok.encode(text, add_special_tokens=False)

    def next_user(self, obs_json):
        text = f"\n<|im_start|>user\n{obs_json}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return self.tok.encode(text, add_special_tokens=False)


def _obs_json(obs):
    return obs.model_dump_json(exclude={"metadata"})


def _append(ep, toks, from_model, logprobs=None):
    """Extend an episode's completion; env tokens get mask 0 and logprob 0.0."""
    ep["completion_ids"] += toks
    ep["env_mask"] += [int(from_model)] * len(toks)
    ep["logprobs"] += logprobs if logprobs is not None else [0.0] * len(toks)


def run_episodes(generate, renderer, jobs, max_turn_tokens, context):
    """Play episodes in lockstep, one batched generate() call per turn.

    Args:
        generate (`Callable`):
            Takes a list of token-id lists and returns one `(token_ids, logprobs)` pair per input.
        renderer ([`Renderer`]):
            Chat renderer for the policy's tokenizer.
        jobs (`list[tuple[str, int]]`):
            `(task_id, seed)` per episode.
        max_turn_tokens (`int`):
            Generation cap per reply.
        context (`int`):
            Token budget for the whole conversation; an episode that cannot fit another reply is finished.

    Returns:
        `list[dict]` with `prompt_ids`, `completion_ids`, `env_mask` (1 on model tokens), `logprobs`,
        `reward`, `end`, `turns`, `task_id` and `seed` per episode.
    """
    eps = []
    for task_id, seed in jobs:
        env = ComputeLabEnvironment()
        obs = env.reset(task_id=task_id, seed=seed)
        eps.append({
            "task_id": task_id, "seed": seed, "env": env, "prompt_ids": renderer.first(_obs_json(obs)),
            "completion_ids": [], "env_mask": [], "logprobs": [], "end": None, "turns": 0, "reward": 0.0, "done": False,
        })

    def finish(ep, end, obs=None):
        """Score through the finish action unless the env already ended; runs once per episode."""
        if obs is None or not obs.done:
            obs = ep["env"].step(ComputeLabAction(**FINISH_ACTION))
        ep["end"] = end
        ep["reward"] = float(obs.reward or 0.0)
        ep["done"] = True
        ep["env"].close()

    while True:
        for ep in eps:
            if not ep["done"] and len(ep["prompt_ids"]) + len(ep["completion_ids"]) + max_turn_tokens > context:
                finish(ep, "context")
        active = [ep for ep in eps if not ep["done"]]
        if not active:
            break
        outputs = generate([ep["prompt_ids"] + ep["completion_ids"] for ep in active])
        for ep, (toks, lps) in zip(active, outputs):
            toks = list(toks)
            _append(ep, toks, from_model=True, logprobs=list(lps))
            if not toks or toks[-1] != renderer.im_end:
                _append(ep, [renderer.im_end], from_model=False)
            ep["turns"] += 1
            action, fmt = parse_reply(renderer.tok.decode(toks, skip_special_tokens=True))
            if action is None or action == {"finish": True}:
                finish(ep, "finish" if action else fmt)
                continue
            if set(action) == {"action"} and isinstance(action["action"], dict):
                action = action["action"]
            try:
                act = ComputeLabAction(**action)
            except (ValidationError, TypeError):
                finish(ep, "schema_invalid")
                continue
            obs = ep["env"].step(act)
            if obs.done:
                finish(ep, "submitted" if act.tool == "submit" else "out_of_calls", obs)
                continue
            _append(ep, renderer.next_user(_obs_json(obs)), from_model=False)
    for ep in eps:
        ep.pop("env")
    return eps
