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

<!-- Example:
       - **Image pull timeout on ml-platform queue** (EXP-005)
         Error: `ErrImagePull: context deadline exceeded`
         Resolution: default pull timeout (5 min) too short; add `imagePullPolicy: IfNotPresent`
         and pre-cache image on target node pool, or use smaller base image. -->

- **CUDA-graph compile modes (max-autotune / reduce-overhead) crash the multi-iteration compile-warmup loop with the cudagraphs overwrite guard** (EXP-021)
  Error: `RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run ... call torch.compiler.cudagraph_mark_step_begin() before each model invocation`
  Resolution: TWO fixes required (EXP-021 Runs 1–2 each found one): (a) `torch.compiler.cudagraph_mark_step_begin()` as the first statement of each warmup iteration AND before the forward in the timed training loop (~µs); (b) `optimizer.zero_grad(set_to_none=True)` INSIDE the warmup loop after each backward — cross-iteration grad accumulation into the previous backward's cudagraph-pool outputs trips the same guard even with (a). Required for ANY torch.compile mode that enables cudagraphs with this train.py warmup pattern.

- **Host CPU-load contamination inflates CHARGED step time with GPU 0 completely free — load avg 241 on the 180-core box ballooned dt 22→72–102ms** (EXP-032)
  Error: `CONTENTION_KILL: 4 windows >27ms, step counter near-frozen; post-kill nvidia-smi shows 0 compute apps / 0% util; uptime 1-min load 241`
  Resolution: charged dt includes H2D + pinned-memory + host-side loop, all of which starve under CPU oversubscription. The GPU-0 zero-compute-apps pre-check is NOT sufficient. Before any launch, also gate on host load: `awk '{print $1}' /proc/loadavg` must be < ~60 (run needs ~10 of 180 cores); poll every 30s until both gates pass. Rerun contaminated runs byte-identically once both gates clear.

- **Foreign process can land on GPU 0 mid-run and silently time-slice it — no exception, just ~2x step time for the contended stretches** (EXP-011)
  Error: `no error — 89 epochs vs ~135 expected; window-avg dt alternating 24ms/48ms; foreign PID at 91% util on GPU 0 post-run`
  Resolution: before launch, confirm GPU 0 has zero compute apps; during the run, detect via THROUGHPUT (pct_done deltas between step prints > ~30ms/step ⇒ contention), NOT printed dt (samples 2% of steps) and NOT PID matching — nvidia-smi reports host-namespace PIDs ([Not Found] names; container pgrep can never match them, our own process gets flagged foreign, EXP-011 Run 3). Host-side load from a GPU-1 job can also inflate step time with no GPU-0 compute app visible (EXP-011 Run 2). Kill+relaunch contaminated runs; never analyze them. Recurred EXP-014 Run 1: a window confirmed-free at launch CLOSED mid-run (evicted job came back). Robust pattern: launch via a composite background script with an INLINE watchdog (sample windowed dt every 15s, auto-kill on 4 consecutive >30ms) — a Monitor armed in a separate agent turn can first poll an already-finished run (turn-scheduling delay) and never fire. Recurred EXP-058 Run 1: episodes BELOW the 4-window kill streak (28/27/30ms spikes) still stole ~500 steps; the binding post-hoc gate is the STEP LEDGER (num_steps below the family band ⇒ contaminated, relaunch byte-identically, never analyze — even with rc=0 and a plausible metric).

- **CPU data-loader margin is ~3%: any added per-image transform cost converts to loader stalls that bust the 600s wall cap without touching the timed budget** (EXP-013)
  Error: `no error — total_seconds 626.9 > 600 cap; epochs unaffected (139) because fetch stalls land outside dt; stalls grew ~50s → ~197s from one cheap PIL op`
  Resolution: the EXP-004 "workers absorb PIL cost" precedent expired when compile (EXP-006) raised GPU demand. Before running any idea that adds per-image CPU work, estimate stall growth against the ~120s wall-clock headroom (baseline total 480.8s); prefer GPU-side/tensor ops or skip the idea.

- **Mid-run `requires_grad_(False)` on module params is a SILENT NO-OP under torch.compile — no recompile, no backward saving, no error** (EXP-055)
  Error: `no error — freeze flip executed but post-flip step time unchanged (22ms, expected ~17.5); startup 12.5s shows the "frozen variant" warmup also compiled nothing (FX cache hit only)`
  Resolution: the cached graph does not guard on param requires_grad here, so flipping it mid-run changes nothing in the compiled fwd/bwd. To change train-time graph topology mid-run, make it graph-visible: a bool attr on the module read in forward (dynamo guards it) gating `tensor.detach()` (cuts backward at the boundary), and pre-warm BOTH flag values in the uncharged warmup so the flip hits a cached variant. Verify engagement via the watchdog dt drop, not via the flip print.

- **torch.compile automatic-dynamic-shapes silently merges multiple warmed batch shapes into ONE dynamic graph ~18% slower at EVERY shape — no warning, no recompile message at run time** (EXP-059)
  Error: `no error — probe P512 26.4ms vs 21.7 family band after warming a second shape (1024); identical read at load 57 and load 37 ruled out host contamination`
  Resolution: any train.py variant feeding >1 batch shape through a compiled model MUST use `torch.compile(model, dynamic=False)` (one static graph per shape) and pre-warm EVERY shape 3 iters in the uncharged warmup. Verify with a two-shape GPU probe whose P512 anchor must sit in [21.5, 23.5] before launch (probes themselves gate at host load < 40 — 40-step samples inflate ~17% at load ~57 even though full runs tolerate up to 60).

### Warning

- **`pkill -f <pattern>` kills the invoking shell itself when the pattern appears in the shell's own command line (exit 144, rest of compound command lost)** (EXP-056)
  Error: `pkill -f exp046_composite inside a bash -c whose eval'd command text contains "exp046_composite" — the cleanup command died mid-script`
  Resolution: kill stray processes by explicit PID (pgrep first, then `kill <pids>` in a separate command), or use a pattern that cannot match your own command line (e.g., `pkill -f '/tmp/exp046_composite\.sh$'` still risks it — PID-based kill is the safe form). Also: never launch the composite detached (`& disown`, stdout to /dev/null) — always Bash run_in_background so gate/watchdog telemetry and a task handle exist., forcing a CIFAR-10 re-download that inflates the NEXT run's startup/total_seconds** (EXP-015)
  Error: `no error — run.log opens with download progress lines; startup_seconds grows by the download time; cap-bust risk if download > ~115s`
  Resolution: always exclude it: `git clean -fd -e .autoresearch/ -e data/`. If a run did pay a download, judge its total_seconds with the startup inflation in mind (download is outside the timed budget and not the experiment's cost — rerun with cached data if the 600s cap busted from download alone).

<!-- Example:
       - **WandB sync delay causes missing early metrics** (EXP-003)
         Error: no error — first 50 steps of metrics missing from WandB dashboard
         Resolution: call `wandb.log(..., commit=True)` at step 0 and every 10 steps during warmup. -->
