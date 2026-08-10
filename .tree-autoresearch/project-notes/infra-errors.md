# Infrastructure Errors

<!-- Cross-loop record of environment, deploy, and job-platform errors and their resolutions.
     Cross-goal by nature (shared machine / cluster / image). Approach-level failures belong in
     goals/{slug}/03-experiment-learnings.md § Failed Approaches — not here.

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

<!-- Example:
       - **Image pull timeout on ml-platform queue** (EXP-005)
         Error: `ErrImagePull: context deadline exceeded`
         Resolution: default pull timeout (5 min) too short; add `imagePullPolicy: IfNotPresent`
         and pre-cache image on target node pool, or use smaller base image. -->

- **GPU allocation harness retained per-step loss tensors** (EXP-017)
  Error: `AssertionError: final_allocation == baseline_allocation`
  Resolution: accumulate finiteness into one fixed device scalar, discard each loss tensor, synchronize, and collect the allocation baseline only after all persistent optimizer/audit state exists.

### Warning

<!-- Example:
       - **WandB sync delay causes missing early metrics** (EXP-003)
         Error: no error — first 50 steps of metrics missing from WandB dashboard
         Resolution: call `wandb.log(..., commit=True)` at step 0 and every 10 steps during warmup. -->

- **Shared `/tmp/__pycache__` can be non-writable** (EXP-014)
  Error: `Permission denied: /tmp/__pycache__/...pyc`
  Resolution: set `PYTHONPYCACHEPREFIX` to an experiment-owned directory under `/tmp` before compiling or running transient harnesses.

- **Compute-app query can retain stale `[Not Found]` GPU contexts** (EXP-021)
  Error: `nvidia-smi --query-compute-apps` reported PID 80142 / 3,384 MiB, while `ps` and `pmon` showed no active process and utilization was 0%.
  Resolution: record the raw context list, but cross-check active contamination with `nvidia-smi pmon`, UUID, and utilization before rejecting a paired benchmark.
