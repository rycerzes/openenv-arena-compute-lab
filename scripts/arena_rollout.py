# SPDX-License-Identifier: BSD-3-Clause

"""Arena-style multi-turn rollouts shared by the local training pilots.

Mirrors the Arena trainer's protocol: a system message with the action schema,
the reset observation as the first user turn, each later observation as the next
user turn (JSON), and one JSON object per reply. Any other reply ends the episode,
which is then scored through the finish action. Qwen3 chat format, thinking off.
"""

import os
import sys

from pydantic import ValidationError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import FINISH_ACTION, SYSTEM, parse_reply  # noqa: E402

from compute_lab.models import ComputeLabAction  # noqa: E402
from compute_lab.server.compute_lab_environment import ComputeLabEnvironment  # noqa: E402

FORMAT_FAILURES = ("bad_json", "no_json", "not_object", "extra_text", "schema_invalid")


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
