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

- **Shared-node GPU contention invalidates the wall-clock-dt-gated budget** (EXP-038, recurred EXP-056)
  Error: per-step dt inflated 8ms → 24–38ms (and epoch count collapsed 91 → 52) because another
  user's jobs (Protenix `batch_inference.py`) were co-located on the same H20; both GPU 0 and 1
  cycled in/out of saturation every few seconds. A run launched during contention under-trains
  and produces a confounded (invalid) regression — NOT a real result. EXP-056 recurrence: a foreign
  proc (PID 194920, 1.4GB, 85% util) grabbed GPU 0 ~seconds AFTER launch (it was idle at check time)
  + CPU load avg ~8 starved the dataloader → wall/Σdt ≈ 10× (the tell: wall 384s vs only 36s Σdt at
  step 2450). Caught and relaunched clean on idle GPU 1 (no epochs wasted on the bad run).
  Resolution: the budget gates on Σ(per-step dt), so a fair run REQUIRES an uncontended GPU.
  Before launching: `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`
  and pick a GPU with util ~0% and mem <700MiB; also check `nvidia-smi --query-compute-apps=pid,used_memory`
  for foreign processes. Then verify dt empirically in the first ~45s (clean baseline-k4 ≈8ms).
  Because free GPUs get grabbed within seconds, use an automated launcher that (a) polls for an idle
  GPU, (b) launches, (c) early-aborts if contended by 90s (too few epochs accrued), (d) accepts only
  a completed run whose num_epochs matches the clean dt (e.g. ≥~85 for baseline-k4 at 8ms). Pattern
  used in EXP-038; reuse it for every dt-budgeted run on this node. `CUDA_VISIBLE_DEVICES=<free_idx>`
  selects the GPU (any of the identical H20s is a fair substitute for device 0).

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

- **A 2nd input resolution under `torch.compile(mode="reduce-overhead")` creates a 2nd CUDA-graph that RAISES the original-shape dt (8→10ms) — multi-shape compile is net-negative on this launch/graph-bound net** (EXP-066)
  Error: no crash — progressive resolution (24×24 then 32×32) gave a 6ms low-res dt but the 32×32-tail dt rose 8→10ms (+25%), cancelling the saving → 89 ep < 91 baseline, and total_seconds 617.2 > 600 (the 2-graph compile/recapture + per-step `F.interpolate` added wall).
  Resolution: do NOT feed >1 input shape to the reduce-overhead-compiled model — each distinct shape gets its own CUDA-graph and the multi-graph state inflates per-step dt for ALL shapes (the conv-dt floor under reduce-overhead is a SINGLE-shape floor). Keep the compiled forward's INPUT shape constant for the whole run (as it is at baseline). Any resolution/shape schedule, test-time multi-crop into the compiled model, or variable-shape batching will pay this penalty. If a shape change is unavoidable, expect to lose the ~1.03× compile benefit and budget wall accordingly.
- **The AugMix-p0.5 base recipe is wall-tight and breaches the 600s budget on run-to-run variance ALONE (no eval-side overhead needed) → RECURRING (3 breaches)** (EXP-061 604.6s change-caused; EXP-065 602.5s base-variance; EXP-066 617.2s change-caused by 2-cudagraph+interpolate, most severe)
  Error: no crash — total_seconds > 600 (budget breach) while training_seconds = 300.0 (the gated COMPUTE budget respected exactly). The wall overrun lives ENTIRELY in eval + dataloader (92 evals + AugMix CPU augmentation variance in the NUM_WORKERS=8 workers), OUTSIDE the Σdt timer. **EXP-061**: an eval-side change (clean-BN recalib, +16 fwd batches/ep + an extra epoch) tipped 593s→604.6s. **EXP-065**: a compute-FREE change (LABEL_SMOOTHING 0.1→0.15, zero throughput impact, identical 8ms dt, 92 ep) STILL hit 602.5s — pure base-recipe variance, the change was not causally involved. EXP-054 baseline 593s with only ~7s margin and high variance (goal-learnings: "a replication may exceed 600s").
  Resolution: (1) Σdt=300s being respected does NOT guarantee total wall < 600s — the eval+dataloader tail has ~±10s variance that can breach on its own. (2) For VERDICT purposes (per EXP-061/065 precedent): when a compute-free or eval-neutral change breaches by a small margin (<~5s) while training_seconds=300.0 is exact and the metric is trustworthy, the metric condition governs → no-improvement, NOT invalid (the breach is base-recipe variance, not a change-caused resource violation). (3) For PLANNING: any experiment adding wall-time OUTSIDE the Σdt timer (eval-side recompute, second loaders, extra forward passes) must budget the 600s wall explicitly — project base wall at the early gate to ≤ ~560s before adding overhead, keep added overhead < ~2s, and prefer testing wall-adjacent ideas on a recipe WITH wall margin (the AugMix recipe has none). Shrink overhead (fewer recalib batches / final-epoch-only) when unavoidable.

- **A data-dependent `if self.training` branch / variable forward output (tuple vs tensor) silently breaks `torch.compile(mode="reduce-overhead")` CUDA-graph capture → dt ~doubles (8→14ms)** (EXP-042)
  Error: no crash — steady-state dt jumped 8→14-16ms (interleaved from ep1), collapsing epochs 91→55 and the metric (under-train); looks like contention but the GPU was idle.
  Resolution: keep the compiled forward's output STRUCTURE static. To add a train-only auxiliary output, do NOT branch the single `forward`; instead define a separate method (e.g. `forward_train`) that ALWAYS returns the same structure `(main, aux)` and compile THAT for the training loop, while the plain `forward` stays byte-identical to baseline (single tensor) for the eager eval handle. Restored dt to 8ms. Always check the dt distribution (`tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c`) after any forward-signature change.
