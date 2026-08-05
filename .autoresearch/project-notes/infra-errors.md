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

### Warning

<!-- Example:
       - **WandB sync delay causes missing early metrics** (EXP-003)
         Error: no error — first 50 steps of metrics missing from WandB dashboard
         Resolution: call `wandb.log(..., commit=True)` at step 0 and every 10 steps during warmup. -->

- **External job on GPU 1 halves throughput → time-budgeted runs fit fewer epochs (confounds absolute accuracy)** (EXP-010, EXP-014, EXP-021)
  Error: no crash — `nvidia-smi` shows a foreign multi-GB job at 100% util sharing GPU 1; throughput drops (~9.8k–12.6k vs ~25–29k img/s) so the 300s budget fits far fewer epochs → under-annealed, NOT comparable to baseline. EXP-010: foreign PID 971585 ~18.5 GB → 59–64 ep. EXP-014 RECURRENCE: foreign PID 1723342 appeared MID-RUN, ramped (14→20+ GB), UNEQUALLY slowing cells (127/74/64 ep). EXP-021 RECURRENCE (AGGRESSIVE): GPU 1 was borrowed by CHURNING foreign jobs (PIDs 2643063→2724456→2807153, each ~10–12 GB/100%), a NEW one landing within ~1 min of each clear window; the first compiled run hit ~12.6k img/s (half clean → ~76 ep) at MAX clocks (1980 MHz — pure contention, not throttle). Note: a persistent idle holder (PID 1723342, ~3.8 GB, 0% util) is TOLERABLE — EXP-014 & EXP-021 ran clean with it present; only ACTIVELY-computing foreign jobs hurt.
  Resolution: check `nvidia-smi` GPU-1 occupancy AND printed `num_epochs`/img/s vs the clean band (uncompiled ~142–154 ep / ~25k img/s; compiled ~173 ep / ~28–29k img/s) before trusting numbers. Under AGGRESSIVE churn, a manual re-run is not enough — use a RETRY-UNTIL-CLEAN ORCHESTRATOR (see `/tmp/exp021_orchestrate.sh`): (1) wait until no foreign compute-app >5GB on the GPU-1 UUID AND util<30 for 2 checks, (2) launch the same-session pair, (3) monitor every ~20s for a foreign >5GB job OR steady img/s<20000 and ABORT+retry the WHOLE pair on either. It caught a clean window once the foreign job exited (c0 back to the 173-ep compiled band). GPU 0 always busy; GPU 1 intermittently but sometimes aggressively borrowed.
