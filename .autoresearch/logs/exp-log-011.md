# EXP-011: EMA weight averaging for evaluation (decay 0.995)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-011.md
- **Plan**: plans/plan-011.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-011
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (verification necessary condition 2 failed — best_test_acc 96.46 < 96.81; clean valid run, research no-improvement)

## Implementation Notes

### Summary

Implemented exactly per plan-011 Milestone 1, all in train.py on branch `autoresearch/exp-011`: (1) `import copy` added to stdlib imports; (2) `EMA_DECAY = 0.995` constant added to the hyperparameter block; (3) after the optimizer construction, an eager EMA copy is created via `copy.deepcopy(base_model).eval()` with all params `requires_grad_(False)`, and param/buffer lists are cached for both the EMA copy and the live eager `base_model` (which shares storage with the compiled `model`); (4) after `optimizer.step()` and before `torch.cuda.synchronize()` — i.e., inside the timed region, honest dt accounting — `torch._foreach_lerp_(ema_params, live_params, 1.0 - EMA_DECAY)` plus a per-buffer `.copy_()` loop for the ~40 BN buffers (running_mean/var float32 + num_batches_tracked int64; direct copy is dtype-safe where lerp is not); (5) the per-epoch eval switched from `base_model` to `ema_model`, still exactly once per epoch through the frozen `Eval`. `uv run python -m py_compile train.py` passed (SYNTAX_OK).

### Surprises & Discoveries

- GPU 0 was occupied at launch time by a foreign process (PID 1987359, `[Not Found]` in this namespace, ~13GB, 86% util) — first time this constraint has actually bitten in this project. Per the hard constraint, waiting for GPU 0 rather than using GPU 1; Monitor watcher polls until the GPU is free.

### Decisions

- Buffer sync uses a plain per-tensor `.copy_()` loop instead of `torch._foreach_copy_`: avoids any foreach dtype edge case with the int64 `num_batches_tracked` buffers, and the cost over ~40 tiny tensors is negligible. (Anticipated in plan § Code Changes.)
- `ema_model` is created AFTER the optimizer but its position relative to the compile-warmup block is irrelevant: warmup does fwd+bwd only, never `optimizer.step()`, so weights are identical either way (plan § Code Changes note).

## Experimental Adjustments

- **Run 2 (retry 1/2, infrastructure) launches with a contention watchdog**: Run 1 was silently time-sliced by a foreign process that arrived ~1 min in; a Monitor polling `nvidia-smi --query-compute-apps` for non-our PIDs on GPU 0 every 20s lets us kill+relaunch within a minute instead of losing a full 7-minute run. Code is UNCHANGED — the failure was environmental; clean-regime windows showed the EMA tax (~1.5ms/step) exactly on budget. (ref: Run 1 — Errors & Dead Ends 2026-06-10; infra-errors.md § Important)

## Run Log

### Run 1

Metadata:
- **Job ID**: background task b8lilmxl0 (local, GPU 0 via CUDA_VISIBLE_DEVICES=0)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log
- **WandB**: N/A
- **Status**: failed (infrastructure — external GPU contention; result NOT a valid hypothesis test)
- **Started**: 2026-06-10 06:29
- **Ended**: 2026-06-10 06:37 (exit 0, 428.0s total)

Description:
- Single run of the baseline-1990397 recipe with one change: per-epoch evaluation scores an EMA (decay 0.995) of the weights instead of the live weights. Training dynamics, schedule, augmentation, and compile are byte-identical to baseline; the EMA update (fused foreach lerp + buffer copy) runs inside the timed region. Expected: dt ≤ 25ms (~133–139 epochs), near-random EMA evals for the first ~2 epochs (init-dominated average — expected, not a bug), trajectory converging onto/above the raw trajectory in the final low-LR epochs, best_test_acc ≥ 96.81 if the hypothesis holds.

Observations:
- GPU 0 busy at launch attempt (foreign PID 1987359, ~13GB, 86% util — `nvidia-smi` 2026-06-10); waited ~25 min per hard constraint "always GPU 0; otherwise wait for it to free up"; freed at 06:29:10 (watcher b98xpp0uy), launched immediately.
- Epoch-1 EMA eval 10.24% — near-random as PREDICTED in plan (EMA still ~61% init weights after 97 steps; weight-space blend of random+trained nets scores ~random). Not a bug. (source: run.log "eval ep   1" line)
- Epoch-5 EMA eval 47.38% — clears the 10.5% bug gate; param/buffer tracking confirmed working. (source: run.log "eval ep   5" line)
- dt gate: 22ms at steps 700–750, byte-identical to baseline cadence — the foreach-lerp + buffer-copy tax is below print granularity; projects ~139 epochs. (source: run.log step lines 700/750)

- Run completed exit 0 but CONTAMINATED: best_test_acc 95.85% on only 89 epochs / 8618 steps — average step cost 34.8ms vs printed dt 22ms. Window analysis (pct_done deltas between step prints) shows long alternating stretches of ~24ms and ~48–54ms window-average — the 2x signature of GPU time-slicing. (source: run.log step lines, summary block)
- Root cause confirmed live: a NEW foreign process (PID 2125435, 6.3GB, 91% util) is on GPU 0 immediately post-run — it arrived mid-run (~step 900) and competed intermittently for the whole run. (source: nvidia-smi --query-compute-apps 06:37)
- Clean-regime windows average ~24ms ≈ baseline 22.3ms + ~1.5ms EMA tax — matches plan budget exactly; the EMA implementation itself is healthy. (source: run.log steps 100–500 window)
- Eval trail still climbing at cutoff (95.60 → 95.85 over the last 8 epochs, final = best) — starvation signature per the plateau-vs-climbing diagnostic, consistent with the epoch deficit, not with an EMA problem. (source: run.log eval lines ep 82–89)
- num_params 4,286,026 = baseline value exactly (plan's "4,292,170" reference figure was a transcription error — all prior logs record 4,286,026; architecture untouched). startup 10.9s = warm inductor cache (also seen EXP-009/010).

Key Metrics:
- best_test_acc: 95.85% @ ep 89 (INVALID for verdict — contaminated run) (source: run.log summary block)
- num_epochs: 89 | num_steps: 8618 | training_seconds: 300.0 | total_seconds: 428.0 | peak_vram_mb: 1631.0 (+18MB over baseline's 1613 — the EMA copy, as predicted) | num_params: 4,286,026

### Run 2 (retry 1/2 — infrastructure)

Metadata:
- **Job ID**: background task bikm7wccx (local, GPU 0 via CUDA_VISIBLE_DEVICES=0)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log (overwritten per run)
- **WandB**: N/A
- **Status**: failed (infrastructure — contaminated again, different mechanism)
- **Started**: 2026-06-10 06:40
- **Ended**: 2026-06-10 06:47 (exit 0, 448.4s total)

Description:
- Identical code to Run 1 (unchanged train.py), relaunched after GPU 0 freed, with a foreign-PID contention watchdog polling `nvidia-smi --query-compute-apps` every 20s.

Observations:
- Contaminated AGAIN, but the watchdog never fired — no foreign compute app ever appeared on GPU 0. Window analysis: steps ~100–3600 ran at ~42–48ms (burning ~166s of the 300s budget at ~2x cost), then the run went clean at 18–24ms through step 10098. (source: run.log pct_done deltas)
- The slow phase coincided with a heavy foreign job on GPU 1 (33GB, 100% util at launch; gone by 06:56) — implicating HOST-side interference (CPU/launch-path pressure from that job's data pipeline), not GPU-0 compute sharing. Our per-step host work (inductor launch path + EMA's ~40 eager buffer copies) makes dt host-sensitive; when the host is loaded, steps become launch-bound (~48ms). (source: watchdog byvytkrm5 empty; nvidia-smi at 06:40 vs 06:56; load avg 10.5 on 180 cores post-run)
- Result: 105 epochs / 10098 steps, best 96.01% @ ~ep 96–100, final 95.92% — epoch-starved vs the ~135-epoch clean projection; NOT a valid hypothesis test. Tail was a shallow plateau around 95.9–96.0 (closer to converged than Run 1 but still ~30 epochs short). (source: run.log eval lines ep 100–105, summary block)
- Clean-phase windows at 18–24ms re-confirm the EMA tax is ≤ ~2ms/step. peak_vram 1631.0MB, startup 10.5s, params 4,286,026 — all as expected.

Key Metrics:
- best_test_acc: 96.01% @ ep ~96–100 (INVALID for verdict — contaminated run) (source: run.log summary block)
- num_epochs: 105 | num_steps: 10098 | training_seconds: 300.0 | total_seconds: 448.4

### Run 3 (retry 2/2 — infrastructure; CLEAN, valid result)

Metadata:
- **Job ID**: background task b219jproe (local, GPU 0 via CUDA_VISIBLE_DEVICES=0)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log (overwritten per run)
- **WandB**: N/A
- **Status**: completed (clean — this run is the valid hypothesis test)
- **Started**: 2026-06-10 06:57
- **Ended**: 2026-06-10 07:06 (exit 0, 479.9s total)

Description:
- Identical code, launched into a fully idle window (both GPUs free, host load 10/180 cores). Two watchers: log-checkpoint monitor and a NEW throughput detector emitting on any >30ms windowed step time (catches any contention mechanism, GPU or host).

Observations:
- CLEAN throughput end-to-end: 139 epochs / 13391 steps, cumulative avg dt 22.4ms — byte-identical to baseline's 22.3ms; the EMA per-step tax is unmeasurable (~0.1ms; the foreach lerp + 40 buffer copies fully overlap). Throughput detector fired ZERO SLOW events. (source: run.log pct/step windows; monitor bsqdeesjm empty)
- The Run-2 watchdog (still alive) flagged "foreign PID 2400676 on GPU 0" continuously from 06:58 — i.e., from the moment Run 3 started. This is OUR OWN process seen through the host PID namespace: nvidia-smi reports host PIDs (hence every process shows `[Not Found]`), so container-side pgrep can never match them. A 139-epoch run is mechanically impossible under real contention — PID-based contention detection is unreliable here; throughput-based detection is the trustworthy signal. (source: monitor byvytkrm5 events 06:58–07:06 vs run.log cadence)
- RESULT: best_test_acc 96.46% @ ep ~130, final 96.43% — BELOW baseline 96.71 by 0.25pp at IDENTICAL epoch count. Tail is a converged plateau (96.38–96.46 over ep 130–139), NOT climbing — this is a genuine research negative, not starvation. (source: run.log eval lines ep 130–139, summary block)
- final_test_loss 0.1874 — better than the raw-eval baselines (~0.20) — the EMA does improve the loss/calibration, just not the max-over-epochs accuracy metric. (source: run.log summary)
- Mechanism (post-hoc): best_test_acc is a MAX over 139 noisy per-epoch evals (±0.1pp at convergence). EMA trades eval variance for mean — smoothing collapses the upper tail of the max-statistic. Unless EMA lifts the MEAN by more than the noise it removes, best-of-smoothed < best-of-noisy. At this baseline strength the mean lift (if any) was smaller than the harvested variance.

Key Metrics:
- best_test_acc: 96.46% (baseline 96.71, bar 96.81 — miss by 0.35pp) (source: run.log summary block)
- num_epochs: 139 | num_steps: 13391 | training_seconds: 300.0 | total_seconds: 479.9 | startup: 10.3s | peak_vram_mb: 1631.0 | num_params: 4,286,026

## Verification Results

### Conditions Checked

1. **Run completed within budget (≤ 600s total)** — `grep "^total_seconds:" run.log` → 479.9s, summary block present (clean exit 0). **PASS**. (source: run.log summary block; task b219jproe exit 0)
2. **best_test_acc ≥ 96.81 (baseline 96.71 + 0.1pp)** — `grep "^best_test_acc:" run.log` → 96.46. **FAIL** (−0.25pp vs baseline; −0.35pp vs bar). (source: run.log summary block)
3. **Validation at most once per epoch** — skipped per first-failure stop. (Informally compliant: 139 "eval ep" lines = num_epochs 139.)

### Informational Metrics

- Not collected (necessary condition 2 failed). Observed anyway in Run 3 summary: peak_vram_mb 1631.0 (+18MB vs baseline 1613 — the EMA copy), num_epochs 139 (throughput fully preserved), num_params 4,286,026 (unchanged).

## Errors & Dead Ends

### 2026-06-10 — Run 1 contaminated by external GPU contention (infrastructure failure)
- Error: `no exception — silent 36% throughput loss: 89 epochs vs ~135 expected; avg step 34.8ms vs ~24ms clean`
- Root cause: a foreign process (PID 2125435) started on GPU 0 ~1 min into the run and time-sliced the GPU intermittently for its whole duration (window-avg dt alternating 24ms / 48–54ms = 2x slicing signature; process confirmed live on GPU 0 at 91% util immediately post-run).
- Source: run.log step/pct lines + `nvidia-smi --query-compute-apps` at 06:37
- Do NOT retry: treating a contended run's metric as a research result — the single-variable attribution is destroyed. Retry requires an exclusive GPU 0 window AND an in-run contention watchdog so contamination is caught at minute ~1, not at the post-mortem.

## Human Notes

> {Researcher can add comments, corrections, or context here}
