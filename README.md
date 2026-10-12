---
license: bsd-3-clause
pretty_name: Verified Compute Lab
tags:
  - openenv
  - reinforcement-learning
  - tool-use
  - agents
configs:
  - config_name: default
    data_files:
      - split: train
        path: tasks.jsonl
---

# Verified Compute Lab (Workbench v3)

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment for
[OpenEnv Arena](https://huggingface.co/spaces/openenvarena/arena): 16 procedural
task families across the eight Arena domains. Each task brings its own tools,
and every answer is checked by an exact verifier.

Each reset builds a fresh task: a question, a few files, tables or an inbox, and
a tool list in the first observation. The action is always one JSON object:

```json
{"tool": "read", "args": {"path": "policy.md"}}
{"tool": "sql", "args": {"query": "SELECT category, SUM(amount) FROM claims GROUP BY category"}}
{"tool": "edit", "args": {"path": "lib.py", "old": "d -= 10", "new": "d -= 9"}}
{"tool": "submit", "args": {"answer": {"orders": {"SKU-412": 240}}}}
```

The agent gets 12 tool calls. Code and queries run in a sandbox as an
unprivileged user, with CPU, memory and output limits; `read` and `grep` return
whole lines and say where to continue.

| Family | Domain | Tools | What it takes |
|---|---|---|---|
| `code_repair` | software engineering | list, read, edit, write, run_tests | fix seeded bugs; at higher levels some bugs have no failing visible test and must be found from SPEC.md |
| `machine_schedule` | industrial | list, read, python | optimal job order for weighted tardiness, with setups, a maintenance window and cancellations |
| `inventory_reorder` | industrial | list, read, python | today's purchase orders from a reorder-point policy, open POs, pack sizes, MOQs and memo changes |
| `inbox_scheduling` | office | list, read, python, list_emails, read_email | earliest meeting slot from constraints spread across emails, UTC offsets and corrections |
| `expense_audit` | office | list, read, sql (no python) | audit claims against a policy: caps by city tier, receipts, weekends, duplicates, FX, dated amendments |
| `auth_forensics` | cybersecurity | list, read, grep (no python) | first successful login from a brute-force source, with decoys, password-using staff, a rotating /24 and log rotation |
| `subtitle_sync` | media | list, read, python | retime an SRT file from sync points, frame-rate conversion and a scene insert |
| `loan_schedule`, `fx_ledger`, `xnpv_xirr` | finance | list, read, python | amortization with corrections, multi-currency ledgers, XNPV and XIRR |
| `reaction_kinetics`, `projectile_ode`, `stoichiometry` | natural science | list, read, python | fits and Arrhenius, Euler integration with drag and wind, yields |
| `constrained_count`, `graph_paths`, `recurrence_mod` | math | list, read, python | digit DP, shortest-path counting, matrix powers for huge N |

Task ids: `<family>` draws a difficulty from 1-4; `<family>-d<k>` pins difficulty
k. A reset without a seed derives the instance from the task id and a 10-minute
window, so the rollouts of one GRPO group (reset together) share an instance and
their rewards compare the policy rather than the luck of the draw. An explicit
seed always wins.

## Reward

All-or-nothing: an answer earns 1 only when every part is right. Partial credit
let a small policy learn to guess categorical fields instead of working, so it
was removed. `code_repair` keeps a share-of-bugs-fixed reward, since every point
needs a real edit; breaking a function that was correct scores 0. Running out of
tool calls, or `{"tool": "submit", "args": {"answer": {}}}`, scores 0.

## Verification

- Oracles score 1.0 for every family, difficulty and seed checked; agent-view
  solvers (tools and prompt only: sandboxed Python, SQL, grep output, parsed
  emails) solve the new families on 40+ seeds each.
- Quality gates over 100 seeds per task: the answer never appears in the prompt,
  and the most common answer solves at most 10% of instances. The gates removed
  three constant answers (the first inbox slot, the identity job order, count 0).
- 598 tests, including regressions for a sandbox launch that could deadlock
  under threads (no `preexec_fn`) and for reads that used to clip the middle.
- The linux/amd64 image passes `openenv validate --url`, and a WebSocket replay
  of every submitted task passes against an anonymous pull on a native x86 host.

## Calibration

Qwen3.8-27B, thinking off, temperature 1.0, through Hugging Face Inference
Providers, with an Arena-style loop (one JSON object per reply; anything else
ends the episode) and the submitted budgets (12 tool calls, 12,288 completion and
20,480 context tokens). Three episodes per pinned task; the submission trains on
the 30 tasks solved in 1 or 2 of 3, covering all 16 families and eight domains.

The reply-format line was A/B tested on 70 episodes per variant: the plain "Reply
with exactly one JSON object per turn." had 12.9% format failures, while longer
escaping advice did worse (17.1% and 24.3%). Details in `format_ab.json`.

## Local experiments and Arena results (honest nulls)

On one RTX 4090, Qwen3-4B trained with GRPO (LoRA, a custom Unsloth trainer with
shared-weight vLLM, 80 steps) learned its training families but showed no gain
on held-out families beyond noise.

| Arena run | Updates | Private tasks |
|---|---|---|
| v1 (17 finance/science/math tasks) | 140 | 4/40: industrial 2, office 2 (same as untrained) |
| v2 (29 tasks, 14 families) | 110 | 3/40: software 1, industrial 2, office 0 |

Across all public runs, the best single scores come from runs that changed the
model least, and runs with steady gradient signal tend to lose solves the base
model had. v3 shortens episodes and tightens grading; expect a score near the
untrained reference.

## Links

- Image: `ghcr.io/rycerzes/openenv-arena-compute-lab:v3` (linux/amd64)
- Source: https://github.com/rycerzes/openenv-arena-compute-lab
- Dataset: https://huggingface.co/datasets/rycerzes/openenv-arena-compute-lab
- OpenEnv at `86a180ede21e044f7929b9a7783ad83aa67d83a3`, as pinned by the Arena

## Files

- `tasks.jsonl`: one row per pinned task id (64), with its domain, tools, whether
  it is in the submission, its calibration, and a seed-0 example prompt and oracle.
- `source/`: environment source, tests, solvers, trainers and scripts.
- `calibration_summary.json`: per-task results against Qwen3.8-27B.
- `format_ab.json`: the reply-format A/B.
