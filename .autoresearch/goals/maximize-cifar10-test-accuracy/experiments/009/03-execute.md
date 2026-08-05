# EXP-009: Muon optimizer (Newton-Schulz orthogonalized momentum) on the conv weights

## Execution

Overall Status & Info:
- **Created**: 2026-06-28
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-009
- **Commit**: (pending — no commit; no-improvement, code discarded)
- **PR**: (pending — none; no-improvement)
- **Outcome**: completed (valid run; result is a no-improvement — see Verification Results)

## Implementation Notes

### Summary
Implemented per plan, all edits in `train.py` (Milestone 1). Added (a) `zeropower_via_newtonschulz5(G, steps=3, eps=1e-7)` — bf16 quintic NS, coeffs (3.4445,−4.7750,2.0315), transpose-if-tall; (b) `class Muon(torch.optim.Optimizer)` — nesterov-momentum buffer → reshape `[out,-1]` → NS-orthogonalize → **airbench weight-renorm** `p·√out/(‖p‖+1e-7)` → `p −= lr·update`, **no weight decay**; (c) constants `PEAK_LR_MUON=0.24`, `MUON_NS_STEPS=3`. In `main()` split learnable params (`p.ndim==4`→Muon, else→SGD), built `muon_opt` + the unchanged EXP-008 `sgd_opt` (lr 0.4, mom 0.9, wd 5e-4) over the non-conv params, added an optimizer-summary print. In the loop: shared triangular `frac`, `lr_muon`/`lr_sgd` set per group, zero_grad+step on both, and the **MUST-FIX** step-50 print updated `lr:`→`lr_m:/lr_s:` (the old `lr` var was removed; this would have crashed at step 50 — plan-review #4). EMA path untouched. Smoke test passed on GPU 1 (singular values 0.83–1.20, no blow-up; renorm gives |p|=2.925≈√8). Scope verified: `git status --porcelain` = only `M train.py`; prepare.py byte-unchanged.

### Surprises & Discoveries
- Newton-Schulz on **random Gaussian** inputs already lands singular values in [0.83,1.20] at just 3 steps (smoke) — better-conditioned than the plan conservatively assumed; real structured gradients should converge at least as well. The `<2.0` blow-up guard had wide margin.
- `from train import ...` triggers the module-level `evaluator = Eval()` (train.py:31) which downloads/loads CIFAR — adds a few seconds to the smoke but is harmless.

### Decisions
- Followed the airbench-grounded design from the plan exactly (conv-only Muon, weight-renorm replacing L2, LR 0.24, ns_steps 3, fc on SGD) — these were the plan's deliberate, review-driven deviations from the idea-03 sketch; no new implementation-time deviation was needed.
- Kept Muon momentum at the shared `MOMENTUM=0.9` (not airbench's 0.6): NS normalizes update scale so momentum only affects smoothing, and 0.9 is the proven value for our long (~150-epoch) schedule and keeps momentum unchanged vs EXP-008.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (local PID — see run_exit sentinel)
- **Log file(s)**: run.log (project root; gitignored)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-28
- **Ended**: 2026-06-29

Description:
- Official EXP-009 run: `CUDA_VISIBLE_DEVICES=1 uv run train.py` under `timeout 600`, full 300s training budget, seed 42 unchanged. Tests whether the airbench Muon package on the conv weights beats EXP-008's tuned SGD-Nesterov (baseline 96.38) by ≥0.10pp (bar 96.48). Expect ~142–150 epochs at ~25k img/s, EMA-evaluated, with the pre-registered trajectory diagnostics read regardless of the final number.

Observations:
- **Divergence-and-recovery trajectory (NOT healthy)**: ep1 71.90 → ep3 77.74 (best), then COLLAPSES to ~random through the entire high-LR phase — ep5 67.90, ep10 53.62, ep25 **10.00%** (= random), ep50 12.41, ep75 15.65, ep100 19.52 — then RE-LEARNS only as the LR anneals: ep120 76.59 → ep138 94.11 (best==final, still rising). 25 epochs had >0.5pp test-acc drops. (source: run.log, `grep "eval ep"`)
- This is the **Muon peak LR TOO HIGH** signature (per plan trajectory diagnostics): orthogonalized + weight-renormed conv updates at lr 0.24 destabilize the net during the high-LR plateau (≈24% conv rotation/step at peak); it recovers only once the LR decays in the tail. Architecture/data pipeline are sound — the clean late recovery to 94.11 proves the collapse was purely an LR/stability problem, not a bug.
- Throughput healthy: img/s steady ~23.4–23.7k (≈EXP-008's band; ns_steps=3 overhead negligible), so the 138 epochs (just below the 142–150 band) is mild host contention, NOT Muon overhead — but moot given the divergence dominates.
- Param split correct: Muon(10 conv tensors) + SGD(22 tensors); num_params 7,784,627 unchanged; peak_vram 1635 MB. (source: run.log L4, summary)

Key Metrics:
- best_test_acc: **94.11%** @ ep138 (= final; baseline 96.38, bar 96.48) (source: run.log summary)
- num_epochs: 138; num_steps: 13378; training_seconds: 300.0; total_seconds: 440.4 (source: run.log summary)
- peak_vram_mb: 1635.3; num_params: 7,784,627 (source: run.log summary)
- collapse depth: test_acc 10.00% @ ep25 (pure random) (source: run.log `eval ep 25`)

## Verification Results

### Conditions Checked

- **NC1 — completes within budget, valid metric, no crash, ≤10 min wall**: PASS. RUN_EXIT=0 (timeout exit ≠124); summary printed valid `best_test_acc: 94.11%`; `training_seconds=300.0` (budget-used hard check 299≤300≤301 PASS); total wall 440.4s < 600s. (source: run.log summary, run_exit sentinel)
- **NC2 — improvement ≥ +0.10pp over baseline (≥96.48)**: **FAIL**. best_test_acc 94.11 < 96.48 (−2.27pp vs baseline 96.38). Anti-bookkeeping check PASS (max per-epoch trace 94.11 == summary best 94.11 — no tampering). → experiment is a **no-improvement**; remaining condition (NC3 scope) not gating but recorded below for completeness.
- **NC3 — genuine/in-scope (recorded, not reached as gate)**: `git status --porcelain` = only `M train.py`; `prepare.py` byte-unchanged; `num_params`=7,784,627 (architecture untouched); seeds intact; ≤1 eval/epoch. No scope or integrity issue — the result is a true no-improvement, not invalid.

### Informational Metrics
- peak_vram_mb: 1635.3 MB (source: run.log summary)
- num_epochs: 138 / num_steps: 13378 / training_seconds: 300.0 (source: run.log summary)
- num_params: 7,784,627 (unchanged) (source: run.log summary)

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
