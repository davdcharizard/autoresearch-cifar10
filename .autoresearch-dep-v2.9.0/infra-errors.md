# Infrastructure Errors

<!-- Cross-loop record of environment, deploy, and job-platform errors and their resolutions.
     Cross-goal by nature (shared machine / cluster / image). Approach-level failures belong in
     goal-learnings/{slug}.md § Failed Approaches — not here.

     Severity tiers:
       Critical  — Blocks execution entirely or causes data loss.
       Important — Causes experiment failure but has a known workaround.
       Warning   — Non-fatal. Degrades performance or produces unexpected behavior.

     Entry format (3-line, budget-strict; HARD CEILING ~500 chars per bullet):

       - **{Concise error — ≤150 chars}** ({source ref})
         Error: `{error message or traceback summary — one line}`
         Resolution: {step-by-step fix — ≤2 lines. Link to issue or commit if applicable.}

     De-duplication: before appending, scan for a matching error signature. If found, extend
     the source-ref list — do not duplicate. -->

## Infrastructure Errors

### Critical

<!-- Example:
       - **OOM on 4-GPU DDP with batch size > 64** (EXP-002)
         Error: `CUDA out of memory. Tried to allocate 2.40 GiB`
         Resolution: reduce per-GPU batch size to 32 or enable gradient checkpointing via
         `model.gradient_checkpointing_enable()`; if still OOM, switch to 8-GPU resource queue. -->

### Important

- **Base conda python lacks torchvision — use `uv run`** (EXP-012, EXP-018)
  Error: `ModuleNotFoundError: No module named 'torchvision'`
  Resolution: Always use `uv run python train.py` instead of bare `python train.py`. The base conda env at `/SPXvePFS/share/miniconda3/bin/python` (Python 3.13.9) does not have torch/torchvision. `uv run` resolves the correct project environment with torch+torchvision+CUDA.

### Warning

<!-- Example:
       - **WandB sync delay causes missing early metrics** (EXP-003)
         Error: no error — first 50 steps of metrics missing from WandB dashboard
         Resolution: call `wandb.log(..., commit=True)` at step 0 and every 10 steps during warmup. -->
