# SPDX-License-Identifier: BSD-3-Clause

"""Local GRPO pilot with TRL on one GPU (vLLM colocated in sleep mode, LoRA, Liger loss).

`train` runs GRPO on the submission tasks minus two held-out families, with
Arena-style multi-turn episodes. `eval` scores the base model and a trained
adapter on fixed seeds: the training tasks and the held-out families.

Usage:
    python scripts/trl_grpo_pilot.py train --model Qwen/Qwen3-4B --steps 60 --out runs/trl1
    python scripts/trl_grpo_pilot.py eval --model Qwen/Qwen3-4B --adapter runs/trl1/final --out runs/trl1
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arena_rollout import Renderer, add_family_args, check_family_args, eval_row, eval_splits, run_episodes, vllm_completions  # noqa: E402


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def _parse_prompt(prompt):
    """Split a `task_id|seed` dataset prompt into `(task_id, seed)`."""
    task_id, seed = prompt.split("|")
    return task_id, int(seed)


def train(args):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    rng = random.Random(0)
    groups_per_step = args.generation_batch // args.group
    n_prompts = args.steps * groups_per_step
    families = args.train_families  # family task ids: the difficulty is drawn on every reset
    prompts = [f"{families[i % len(families)]}|{rng.randrange(2**31)}" for i in range(n_prompts)]
    os.makedirs(args.out, exist_ok=True)
    episode_log = open(os.path.join(args.out, "episodes.jsonl"), "a")
    state = {"renderer": None}

    def rollout(batch_prompts, trainer):
        if state["renderer"] is None:
            state["renderer"] = Renderer(trainer.processing_class)
        g = trainer.num_generations
        # TRL may pass each prompt already repeated g times; collapse that so every prompt gets exactly g episodes.
        duplicated = len(batch_prompts) % g == 0 and all(p == batch_prompts[i - i % g] for i, p in enumerate(batch_prompts))
        unique = batch_prompts[::g] if duplicated else batch_prompts
        jobs = [_parse_prompt(p) for p in unique for _ in range(g)]

        def generate(token_lists):
            _, completions, logprobs, _ = trainer.vllm_generation.generate(token_lists, None, 1)
            return [(c, [lp[0] if lp and lp[0] is not None else 0.0 for lp in lps]) for c, lps in zip(completions, logprobs)]

        t0 = time.time()
        eps = run_episodes(generate, state["renderer"], jobs, args.max_turn_tokens, args.context)
        rollout_s = round(time.time() - t0)
        for ep in eps:
            row = {k: ep[k] for k in ("task_id", "seed", "reward", "end", "turns")}
            row |= {"step": trainer.state.global_step, "model_tokens": sum(ep["env_mask"]), "rollout_s": rollout_s}
            episode_log.write(json.dumps(row) + "\n")
        episode_log.flush()
        return {
            "prompt_ids": [ep["prompt_ids"] for ep in eps],
            "completion_ids": [ep["completion_ids"] for ep in eps],
            "logprobs": [ep["logprobs"] for ep in eps],
            "env_mask": [ep["env_mask"] for ep in eps],
            "env_reward": [ep["reward"] for ep in eps],
        }

    def env_reward(prompts, completions, env_reward, **kwargs):
        return env_reward

    config = GRPOConfig(
        output_dir=args.out,
        max_steps=args.steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=3,
        per_device_train_batch_size=args.micro_batch,
        generation_batch_size=args.generation_batch,
        gradient_accumulation_steps=args.generation_batch // args.micro_batch,
        num_generations=args.group,
        max_completion_length=args.max_turn_tokens,
        temperature=1.0,
        beta=0.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=True,
        vllm_gpu_memory_utilization=args.vllm_memory,
        vllm_max_model_length=args.context,
        vllm_importance_sampling_correction=False,
        use_liger_kernel=True,
        gradient_checkpointing=True,
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16"},
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(args.steps // 2, 1),
        report_to="none",
        shuffle_dataset=False,
        remove_unused_columns=False,
    )
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=env_reward,
        args=config,
        train_dataset=Dataset.from_dict({"prompt": prompts}),
        peft_config=lora,
        rollout_func=rollout,
    )
    _write_json(os.path.join(args.out, "config.json"), {"splits": eval_splits(args.train_families, args.held_out), **vars(args)})
    trainer.train()
    trainer.save_model(os.path.join(args.out, "final"))
    _write_json(os.path.join(args.out, "log_history.json"), trainer.state.log_history)


def evaluate(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    splits = eval_splits(args.train_families, args.held_out)
    tok = AutoTokenizer.from_pretrained(args.model)
    renderer = Renderer(tok)
    llm = LLM(model=args.model, enable_lora=bool(args.adapter), max_lora_rank=args.rank, max_model_len=args.context, gpu_memory_utilization=0.85, enable_prefix_caching=True)
    params = SamplingParams(max_tokens=args.max_turn_tokens, temperature=args.eval_temperature, seed=0)
    seeds = [10_000 + i for i in range(args.eval_seeds)]
    models = [] if args.no_base else [("base", None)]
    if args.adapter:
        models.append(("trained", LoRARequest("trained", 1, args.adapter)))
    results = {}
    for name, lora in models:
        def generate(token_lists):
            outs = llm.generate([{"prompt_token_ids": ids} for ids in token_lists], params, lora_request=lora, use_tqdm=False)
            return vllm_completions(outs)

        for split, tasks in splits.items():
            t0 = time.time()
            eps = run_episodes(generate, renderer, [(t, s) for t in tasks for s in seeds], args.max_turn_tokens, args.context)
            row = eval_row(name, split, eps, time.time() - t0)
            results[f"{name}/{split}"] = row
            print(json.dumps({k: v for k, v in row.items() if k != "by_task"}), flush=True)
    os.makedirs(args.out, exist_ok=True)
    _write_json(os.path.join(args.out, f"eval_{args.tag}.json"), results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--generation-batch", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--vllm-memory", type=float, default=0.5)
    ap.add_argument("--max-turn-tokens", type=int, default=768)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--eval-seeds", type=int, default=6)
    ap.add_argument("--eval-temperature", type=float, default=0.7)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--no-base", action="store_true", help="eval: skip the base model")
    add_family_args(ap)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    check_family_args(ap, args)
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
