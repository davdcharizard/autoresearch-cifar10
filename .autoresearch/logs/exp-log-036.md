# EXP-036: Periodic / sparse SAM (Sharpness-Aware Minimization on every 5th step, ρ=0.05)

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-036.md
- **Plan**: plans/plan-036.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-036
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Milestone 1 per plan-036: added sparse SAM to baseline `train.py` in three edits, no architecture/data/eval change. (1) Constants `SAM_RHO = 0.05`, `SAM_EVERY = 5`. (2) Hoisted `sam_params = [p for p in model.parameters() if p.requires_grad]` before the loop. (3) Replaced the single-pass step with a two-pass SAM step gated by `do_sam = step % SAM_EVERY == 0`: first forward-backward = clean grad (logged loss); on SAM steps, under `torch.no_grad()` compute global grad-norm, ascend `w += rho*g/||g||` via `torch._foreach_mul/_add_` (vectorized per the EXP-031 throughput lesson), second forward-backward at the perturbed point, restore `w -= e` via `_foreach_sub_`; `optimizer.step()` then uses the SAM gradient (clean gradient on non-SAM steps). Smoke test passed: AST clean, diff = train.py only, and the compiled (reduce-overhead) two-pass step ran 12 steps with SAM firing on steps 5/10 (finite grad-norms 6.4/11.8), no CUDA-graph/dynamo error, all losses finite.

### Surprises & Discoveries
- **torch.compile(reduce-overhead) handled the two-pass SAM step with in-place param perturbation cleanly** — no cudagraph "overwrite"/static-buffer error, so the planned fallback ladder (mark_step_begin → default mode → eager) was NOT needed. In-place `_foreach_add_/sub_` on params behaves like the optimizer's own in-place update for cudagraph replay, as anticipated.

### Decisions
- Kept `reduce-overhead` compile (smoke confirmed compatibility) to preserve the ~8ms launch-bound throughput — essential so the sparse-SAM run stays near-converged (~76 ep) rather than tripping the compute wall.
- Log the clean first-pass loss (standard SAM convention) so the loss curve is comparable to prior experiments.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID — background task)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09

Description:
Full 300s-compute-budget training of the k=4 WideResNet with sparse SAM (ρ=0.05) applied on every 5th step (plain Nesterov-SGD otherwise). Hypothesis: SAM steers SGD toward a flatter minimum that generalizes better, lifting best_test_acc above the 96.22 plateau toward the 96.32 bar, while the sparse cadence keeps epochs ≥~75 so the test is throughput-fair. Expect ~75–78 ep, params 4,299,866.

Observations:
- **SAM ran correctly, no compile/cudagraph issues** — exit 0, 0 Traceback/CUDA/cudagraph/NaN. The reduce-overhead two-pass SAM step worked for the full run (the smoke-test confirmation held at scale).
- **Throughput cost was ~1.27× (not the hoped ~1.0×)**: mean dt = 300000ms/29527 steps = **10.16ms** (vs baseline 8ms), → **76 epochs** (vs baseline 91). This matches the plan's ~76-ep prediction (mid-run printed dt of 8-9ms at SAM steps was misleading — the SAM second forward-backward does cost real GPU time). 76 ep ≥ the ~75 fairness gate, so the test is throughput-fair-ish, but it is ~16% fewer epochs than baseline.
- **Converged, NOT severely underfit**: final_test_loss **0.1969 ≈ baseline 0.195**, and the tail is dead-flat (ep71-76 all 95.84-95.89, peak **95.89 @ ep72**). The 76-ep SAM run reached baseline-quality loss and plateaued — so the −0.33pp top-1 gap is NOT a gross under-training artifact (cf. EXP-022 underfit loss 0.224); SAM's flat-minima mechanism simply did not produce a top-1 gain here.
- params 4,299,866 (unchanged), 76 evals == 76 epochs. (source: run.log summary + eval lines)

Key Metrics:
- best_test_acc: **95.89%** @ ep72 (baseline 96.22, bar 96.32 → **−0.33pp vs baseline, −0.43pp vs bar**)
- final_test_acc: 95.84%; final_test_loss: 0.1969 (≈ baseline 0.195 — converged)
- num_epochs: 76; num_steps: 29,527; mean dt 10.16ms (1.27× baseline 8ms)
- num_params: 4,299,866 (unchanged ✓); peak_vram_mb: 453.4; total_seconds: 393.3 (<600 ✓)
- (source: run.log summary block)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar (best_test_acc ≥ 96.32)**: **FAIL.** best_test_acc = 95.89% < 96.32 (−0.43pp below bar; −0.33pp below the 96.22 baseline). (source: `grep "^best_test_acc:" run.log`)
- **Cond 2 — clean completion within budget**: **PASS.** Summary printed; `grep -c "Traceback|RuntimeError|CUDA error|cudagraph|NaN|Killed" run.log` == 0; total_seconds 393.3 < 600.
- **Cond 3 — no constraint violations**: **PASS.** `git diff --name-only` == only train.py; num_params 4,299,866 (unchanged); eval-count 76 == num_epochs 76 (≤ once/epoch); core torch only (no new deps); seed 42 unchanged.
- **Fairness/attribution gate**: realized num_epochs 76 ≥ ~75 gate (throughput-fair-ish; mean dt 10.16ms = 1.27×); loss 0.1969 ≈ baseline → converged. No compile fallback used (reduce-overhead worked). Result is trustworthy as a fair-ish test of sparse SAM, with the caveat that it ran 16% fewer epochs than baseline.

### Informational Metrics
- peak_vram_mb: 453.4 (≈ baseline)
- num_epochs / num_steps: 76 / 29,527 (mean dt 10.16ms = 1.27× baseline; SAM second pass cost real time)
- final_test_loss: 0.1969 (≈ baseline 0.195 — converged, not underfit)

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
