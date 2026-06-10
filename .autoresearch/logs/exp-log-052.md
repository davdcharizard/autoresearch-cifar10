# EXP-052: AugMix replacing TrivialAugmentWide (strongest diverse augmentation)

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-052.md
- **Plan**: plans/plan-052.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-052
- **Commit**: 292a9e2 (on autoresearch/exp-052, merged to autoresearch/dev)
- **PR**: N/A — repository is local-only by design (no git remote); commits kept local
- **Outcome**: completed

## Implementation Notes

### Summary
Single-line transform swap in `train.py`'s `train_tf`: `transforms.TrivialAugmentWide()` → `transforms.AugMix()` (torchvision defaults: severity=3, mixture_width=3, chain_depth=-1, alpha=1.0), at the same PIL-stage position (after RandomCrop+Flip, before ToTensor). Everything else (Cutout, model, optimizer, schedule, seed, batch, compile) unchanged. Maps to Milestone 1 of plan-052. Smoke tests passed: AST OK; `git diff --name-only` = train.py only; the diff is exactly the comment + the one transform line; AugMix runs on 5 CIFAR PIL samples yielding (3,32,32) float32; num_params = 4,299,866 (unchanged — augmentation does not touch the model).

### Surprises & Discoveries
None during implementation. AugMix is a drop-in positional replacement for TrivialAugmentWide (both PIL-stage auto-augmentations). The open question is wall-clock: AugMix's 3-chain CPU cost (~3× TA) may starve the GPU on the Σdt budget — monitored at Milestone 2.

### Decisions
Used torchvision AugMix defaults (no severity/mixture_width tuning) to test the canonical strongest-diverse variant first, per the brainstorm. Lighter-AugMix (mixture_width=2, chain_depth=2) is held as a contingency only if Run 1 breaches the wall-clock feasibility gate.

## Experimental Adjustments

- **AugMix defaults → mixture_width=2, chain_depth=1 (Run 1 → Run 2)**: Run 1 (default AugMix w3,d-1) breached the wall-clock feasibility gate — clean intra-epoch wall rate 46.3ms/step vs 8ms GPU dt (5.8× starvation), projecting ~1500–1850s wall (≫600s limit). A direct dataloader-throughput probe (8 workers) confirmed default AugMix = 21.1ms/batch (~792s full-budget wall), and even the plan's contingency w2,d2 = 17.9ms (~670s) — both breach 600s. The most diverse config that fits is w2,d1 = 12.6ms/batch (~572s isolated). Reduced to w2,d1, which preserves AugMix's defining mechanism (mixing ≥2 chains with the clean image) at the lightest viable cost. (ref: Run 1 obs; dataloader probe in conversation log)

## Run Log

### Run 1

Metadata:
- **Job ID**: (pending — local background PID)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: submitted
- **Started**: (pending)
- **Ended**: pending

Description:
- Runs the AugMix-swapped training (`uv run train.py`) on idle GPU 1 within the fixed 300s Σdt budget. Tests whether the strictly more diverse mix-of-chains augmentation regularizes better than single-chain TrivialAugment on this generalization-bound k=4 net (bar 96.32 vs baseline 96.22). Expect dt steady ~8ms (AugMix is CPU-side, GPU step unchanged); the key risk is wall-clock — if AugMix's CPU cost starves the GPU, wall balloons toward the 600s limit. Feasibility checked early.

Observations:
- ABORTED at ~step 3700/ep10 (11.8%, Σdt 35.4s). dt steady 8ms (GPU step unchanged, as predicted — AugMix is CPU-side). Clean intra-epoch wall rate 46.3ms/step → 5.8× GPU-dt starvation → projected wall ~1500–1850s ≫ 600s limit. Default AugMix infeasible on the Σdt budget at 8 workers. (source: run.log; wall-rate measured in conversation log)

Key Metrics:
- test_acc @ ep6: 78.56% (early, mid-warmup — not meaningful) (source: run.log "eval ep 6")

### Run 2

Metadata:
- **Job ID**: (pending — local background PID)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09
- **Ended**: 2026-06-09 (exit 0, 571.9s wall)

Description:
- Re-run with the feasibility-reduced AugMix(mixture_width=2, chain_depth=1) — the most diverse config that fits the 600s wall on 8 workers (12.6ms/batch isolated, ~572s projected). Still tests AugMix's core mechanism (mixing 2 augmentation chains + clean image, strictly more diverse than single-op TrivialAugment). Gated on a real-load early wall measurement: if projected total wall > 585s, abort (isolated probes underestimate under real GPU+worker CPU contention). Expect dt ~8ms; bar 96.32 vs baseline 96.22.

Observations:
- FEASIBILITY GATE PASSED. dt steady 8ms (GPU step unchanged). Real-load eval-inclusive wall rate = 15.2ms/step at step 7350 (21.6%); projected total ~549s (≈519s train+eval + ~30s compile) — under the 600s limit with ~51s margin, clears the 585s abort gate. (source: run.log; wall measured in conversation log). Letting run complete.

Key Metrics:
- best_test_acc: 96.34% @ ep89 (source: run.log "eval ep 89"; summary best_test_acc)
- final_test_acc: 96.25% @ ep91; final_test_loss: 0.2010
- total_seconds: 571.9s (wall); training_seconds: 300.0 (Σdt budget); startup: 2.0s
- num_epochs: 91; num_steps: 35,304; num_params: 4,299,866; peak_vram: 453.8 MB
- dt dist: 625×8ms, 80×9ms, 1×24ms (steady 8ms — GPU step unchanged, AugMix CPU-side as predicted)

## Verification Results

### Conditions Checked
- **Cond 1 — best_test_acc ≥ 96.32 (baseline 96.22 + 0.1)**: 96.34% → **PASS** (+0.12pp vs baseline; clears the +0.1 bar). (source: run.log summary)
- **Cond 2 — clean completion within budget**: summary printed, total_seconds 571.9 < 600 ✓, num_params 4,299,866 ✓, no NaN/traceback/error (grep count 0) → **PASS**. (source: run.log)
- **Cond 3 — no hard-constraint violations**: `git status --porcelain` = ` M train.py` only (eval/prepare untouched); AugMix is torchvision-native (no new dep); seed 42 unchanged; eval loop unchanged (once/epoch); no seed hacking → **PASS**.
- **All necessary conditions PASS → Outcome: completed.**

### Informational Metrics
- delta vs baseline 96.22: **+0.12pp**. num_epochs 91 (= baseline, no epoch loss — w2,d1's CPU cost is dataloader-bound but Σdt budget unaffected). final_test_loss 0.2010 ≈ baseline 0.195 (slightly higher; gain is a top-1 generalization effect, not a loss-polish effect). peak_vram 453.8 MB (= baseline, augmentation is CPU-side).

## Errors & Dead Ends

## Human Notes

> {none — autopilot}
