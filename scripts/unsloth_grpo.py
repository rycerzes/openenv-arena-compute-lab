# SPDX-License-Identifier: BSD-3-Clause

"""Custom single-GPU GRPO trainer: Unsloth LoRA + shared-weight vLLM, Arena-style multi-turn episodes.

Unsloth loads the policy once and serves it through vLLM (`fast_inference`), so
sampling and training share the weights instead of holding two copies. Each step
plays groups of episodes in lockstep (one batched generate call per turn), scores
them with the environment, and takes one policy-gradient step with group-mean
advantages on model-written tokens only. Log-probs are computed from hidden
states in sequence chunks, so full-vocabulary logits are never materialized.

Usage:
    python scripts/unsloth_grpo.py --model Qwen/Qwen3-4B --steps 60 --out runs/u1
    python scripts/unsloth_grpo.py --model Qwen/Qwen3-8B --load-4bit --steps 60 --out runs/u8
"""

import os

os.environ.setdefault("UNSLOTH_RETURN_HIDDEN_STATES", "1")

from unsloth import FastLanguageModel  # noqa: E402  (must load before transformers)

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402
from vllm import SamplingParams  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arena_rollout import Renderer, add_family_args, check_family_args, eval_row, eval_splits, run_episodes, vllm_completions  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _chunk_logp(hidden, weight, targets):
    logits = (hidden @ weight.t()).float()
    return logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def sequence_logp(model, ids, chunk):
    """Log-prob of each token ids[1:] given its prefix, without materializing full logits."""
    input_ids = torch.tensor([ids[:-1]], device="cuda")
    hidden = model(input_ids=input_ids).logits[0]  # hidden states (UNSLOTH_RETURN_HIDDEN_STATES=1)
    weight = model.get_output_embeddings().weight
    targets = torch.tensor(ids[1:], device="cuda")
    parts = [checkpoint(_chunk_logp, hidden[s : s + chunk], weight, targets[s : s + chunk], use_reentrant=False) for s in range(0, len(targets), chunk)]
    return torch.cat(parts)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model, self.tok = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.context,
            load_in_4bit=args.load_4bit,
            fast_inference=True,
            max_lora_rank=args.rank,
            gpu_memory_utilization=args.vllm_memory,
        )
        self.model = FastLanguageModel.get_peft_model(
            self.model, r=args.rank, lora_alpha=2 * args.rank, target_modules=TARGETS,
            use_gradient_checkpointing="unsloth", random_state=0,
        )
        self.renderer = Renderer(self.tok)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        self.params = params
        self.lora = None
        self.sync_dir = os.path.join(args.out, "lora_sync")

    def sync(self):
        """Push the current LoRA weights to the vLLM engine (in memory)."""
        self.model.save_lora(self.sync_dir)
        self.lora = self.model.load_lora(self.sync_dir, load_tensors=True)

    def generator(self, temperature, lora):
        sp = SamplingParams(max_tokens=self.args.max_turn_tokens, temperature=temperature, stop=["<|im_end|>"])

        def generate(token_lists):
            outs = self.model.fast_generate([{"prompt_token_ids": ids} for ids in token_lists], sampling_params=sp, lora_request=lora, use_tqdm=False)
            return vllm_completions(outs)

        return generate

    def rollout(self, jobs, temperature, lora):
        return run_episodes(self.generator(temperature, lora), self.renderer, jobs, self.args.max_turn_tokens, self.args.context)

    def update(self, eps, advantages):
        self.model.train()
        # Dr. GRPO: a constant normalizer, so long episodes are not weighted by their length
        norm = len(eps) * self.args.max_turn_tokens
        self.optim.zero_grad(set_to_none=True)
        for ep, adv in zip(eps, advantages):
            if adv == 0:
                continue
            ids = ep["prompt_ids"] + ep["completion_ids"]
            mask = [0] * len(ep["prompt_ids"]) + ep["env_mask"]
            logp = sequence_logp(self.model, ids, self.args.logit_chunk)
            m = torch.tensor(mask[1:], device="cuda", dtype=logp.dtype)
            loss = -(adv * (logp * m).sum()) / norm
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, 1.0).item()
        self.optim.step()
        self.sync()
        return grad_norm


def evaluate(trainer, splits, seeds, out_path, models):
    results = {}
    for name, lora in models:
        for split, tasks in splits.items():
            t0 = time.time()
            eps = trainer.rollout([(t, s) for t in tasks for s in seeds], trainer.args.eval_temperature, lora)
            row = eval_row(name, split, eps, time.time() - t0)
            results[f"{name}/{split}"] = row
            print(json.dumps({k: v for k, v in row.items() if k != "by_task"}), flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--groups-per-step", type=int, default=2)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--vllm-memory", type=float, default=0.55)
    ap.add_argument("--max-turn-tokens", type=int, default=768)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--logit-chunk", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--eval-temperature", type=float, default=0.7)
    ap.add_argument("--eval-seeds", type=int, default=4)
    ap.add_argument("--skip-base-eval", action="store_true")
    add_family_args(ap)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    check_family_args(ap, args)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    splits = eval_splits(args.train_families, args.held_out)
    seeds = [10_000 + i for i in range(args.eval_seeds)]
    trainer = Trainer(args)
    if not args.skip_base_eval:
        evaluate(trainer, splits, seeds, os.path.join(args.out, "eval_base.json"), [("base", None)])

    rng = random.Random(0)
    log = open(os.path.join(args.out, "train.jsonl"), "a")
    trainer.sync()
    for step in range(1, args.steps + 1):
        t0 = time.time()
        tasks = [args.train_families[(step * args.groups_per_step + i) % len(args.train_families)] for i in range(args.groups_per_step)]
        jobs = [(t, s) for t in tasks for s in [rng.randrange(2**31)] for _ in range(args.group)]
        eps = trainer.rollout(jobs, args.temperature, trainer.lora)
        t_roll = time.time() - t0
        advantages = []
        for g in range(args.groups_per_step):
            group = eps[g * args.group : (g + 1) * args.group]
            mean = statistics.mean(ep["reward"] for ep in group)
            advantages += [ep["reward"] - mean for ep in group]
        grad_norm = trainer.update(eps, advantages) if any(advantages) else 0.0
        row = {
            "step": step, "tasks": tasks, "mean_reward": round(statistics.mean(ep["reward"] for ep in eps), 4),
            "rewards": [ep["reward"] for ep in eps], "ends": [ep["end"] for ep in eps],
            "zero_std_groups": sum(not any(advantages[g * args.group : (g + 1) * args.group]) for g in range(args.groups_per_step)),
            "model_tokens": sum(sum(ep["env_mask"]) for ep in eps), "grad_norm": round(grad_norm, 4),
            "rollout_s": round(t_roll), "step_s": round(time.time() - t0),
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps({k: v for k, v in row.items() if k not in ("rewards", "ends")}), flush=True)
    trainer.model.save_lora(os.path.join(args.out, "final"))
    evaluate(trainer, splits, seeds, os.path.join(args.out, "eval_final.json"), [("trained", trainer.lora)])


if __name__ == "__main__":
    main()
