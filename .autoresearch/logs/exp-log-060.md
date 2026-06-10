# EXP-060: AutoAugment(CIFAR10 learned policy) replacing AugMix

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-060.md
- **Plan**: plans/plan-060.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-060
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Single-line augmentation swap (Milestone 1): replaced `transforms.RandomApply([transforms.AugMix()], p=0.5)` at train.py L171 with `transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)` (full native coverage), in the same Compose slot (after RandomHorizontalFlip, before ToTensor) so it receives PIL images. Updated the comment block to describe the CIFAR10 learned policy and the closed-axis justification. Everything else byte-identical to EXP-054 (k=4 WideResNet-20, GPU Cutout(16), cosine peak0.2/warmup0.05, Nesterov m0.9, WD 1e-4, LS 0.1, batch 128, seed 42, compile reduce-overhead). Smoke: AST OK; `git diff --name-only` = train.py only; 5 PIL samples through the full train_tf → finite (5,3,32,32) float32.

### Surprises & Discoveries
- None at implementation. AutoAugment(CIFAR10) is core torchvision 0.24.1 (verified import + policy enum), so no new dependency. It accepts PIL images directly in the Compose slot.

### Decisions
- **Full native coverage (not RandomApply p=0.5) as the primary config**: AutoAugment was designed/validated at full coverage (one sub-policy per image), and its per-op internal probabilities keep effective per-image strength moderate — so full coverage ≈ moderate effective coverage, unlike full-coverage AugMix (EXP-057) which stacked 3 chains. The p=0.5 RandomApply wrapper is held as an in-flight fallback if the early gate shows over-regularization OR a wall-infeasible projection (per plan Milestone 2 / Abort Criteria).
- **GPU 1 chosen**: fully idle (0 MiB, 0%) at pre-launch; GPU 0 had 1043 MiB resident (avoid for a fair dt-budgeted run).

## Experimental Adjustments

- **Kept full-coverage AA (did NOT switch to the p=0.5 fallback)**: the plan's two fallback triggers both cleared at the early gate. (1) WALL: at wall 53s / Σdt 36s (step ~4050, ep11), img/s steady ~15,300 → real per-step ~8.37ms ≈ dt 8ms, i.e. the 8 dataloader workers keep up with no starvation (AutoAugment's ~2 ops/image is far lighter than AugMix-w3's 5.8× starvation in EXP-052). Projected total wall ≈ startup 13s + training ~312s + ~91 evals ~36s ≈ ~361s ≪ 600s. (2) OVER-REG: early acc ep1 46.8→ep5 72.4→ep10 81.2 is a healthy climb; lower-than-EXP-054 early acc is the expected stronger-aug signature, not by itself a worse-final signal — so no premature switch. (ref: Run 1 — run.log step ~4050, ps etimes 53s.)

## Run Log

### Run 1

Metadata:
- **Job ID**: (pending — background bash, GPU 1)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09

Description:
- AutoAugment(CIFAR10 learned policy) at full coverage replacing the EXP-054 CPU AugMix-p0.5, keeping GPU Cutout(16); all else byte-identical. Tests whether the CIFAR-specialized learned auto-aug policy supplies a better diversity distribution than tuned AugMix (96.45), through the same epoch-free CPU delivery. Bar = 96.55. Launched on idle GPU 1. Expected: ~91 epochs at dt ~8ms; the binding risk is the 600s WALL (CPU aug), gated early.

Observations:
- Clean startup: ResNet-20, params 4,299,866, batch/epoch 390, dt steady 8ms (CPU aug epoch-free as designed). No NaN, loss descending. (source: run.log head)
- Early acc trajectory (stronger-aug signature, lower early than EXP-054 but healthy climb): ep1 46.83%, ep2 58.07, ep3 63.36, ep4 66.57, ep5 72.40, ep10 81.17%. (source: run.log `eval ep` lines)
- Wall gate PASSED at ep11 — see Experimental Adjustments (no starvation, ~361s projected, p=0.5 fallback not needed). GPU 0 has an unrelated v2.9.5 job; my run is solo on GPU 1 (no contention).

Key Metrics:
- **best_test_acc: 96.22%** (baseline 96.45, bar 96.55 → **−0.23pp, no-improvement**) (source: run.log `best_test_acc: 96.22%`)
- final_test_loss: 0.1942 (≈ EXP-054's 0.1968 — converged, NOT underfit; best hit ep89) (source: run.log `final_test_loss`, `eval ep 89`)
- num_epochs: 91, num_steps: 35444 (FULL budget — CPU AutoAugment is epoch-free as designed, matching EXP-054's 91 ep) (source: run.log)
- total_seconds: 402.5 (< 600 ✓), peak_vram_mb: 453.8, num_params: 4,299,866 ✓ (source: run.log)
- **Decisive comparison: AutoAugment(CIFAR10) full-coverage 96.22 = TrivialAugment 96.22 (EXP-012), both < AugMix-p0.5 96.45 (EXP-054).** Confirms the TrivialAugment-paper thesis (AA≈TA on CIFAR-10); the CIFAR-learned policy is NOT a richer diversity distribution than tuned AugMix here. Converged loss (0.1942) rules out an epoch/wall artifact — it is a genuine policy-strength result.

## Verification Results

### Conditions Checked

- **Necessary condition 1 — best_test_acc >= 96.55**: actual **96.22** → **FAIL** (−0.23pp). Verdict = no-improvement. Stop at first failed necessary condition. (source: run.log `best_test_acc: 96.22%`)
- (For completeness, not gating:) Condition 2 — clean completion: total_seconds 402.5 < 600 ✓, num_params 4,299,866 ✓, summary block printed ✓, NaN/traceback/error count 0 ✓. Condition 3 — scope: `git diff --name-only` = train.py only ✓; prepare.py/eval untouched ✓; evaluate() once/epoch (unchanged loop) ✓; no new deps (AutoAugment core torchvision 0.24.1) ✓; seed 42 unchanged ✓; ran solo on uncontended GPU 1 (steady 8ms, img/s ~15,300, wall/Σdt ~1.05×) ✓.

### Informational Metrics
- peak_vram_mb: 453.8 (source: run.log)
- num_epochs/num_steps: 91 / 35444 — full epoch budget, identical to EXP-054 (CPU aug epoch-free) (source: run.log)
- final_test_loss: 0.1942 (source: run.log) — slightly below EXP-054's 0.1968, but top-1 lower (loss-not-top1: AA's policy yields confident-but-not-more-accurate predictions vs AugMix)

## Errors & Dead Ends

## Human Notes

> {none — autopilot}
