# EXP-008: Squeeze-Excitation blocks on k=4 (+ torch.compile enabler)

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-008.md
- **Plan**: plans/plan-008.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-008
- **Commit**: (none — no-improvement, changes discarded)
- **PR**: (none — no-improvement)
- **Outcome**: completed (clean run; verification cond 2 failed → no-improvement verdict in analyze)

## Implementation Notes

### Summary
Edits to `train.py` only (Milestone 1): (1) added `SEModule` (GAP → FC(C→C/16) → ReLU → FC → sigmoid → channel
scale); (2) added `self.se = SEModule(out_channels, SE_REDUCTION)` to `BasicBlock` and applied it to the post-bn2
residual-branch output before the residual add; (3) added `SE_REDUCTION = 16`; (4) added
`compiled_model = torch.compile(model, mode="reduce-overhead")` and routed the training forward through it; eval
unchanged on the eager `model`. Parse-clean, ruff clean, param count 4,333,550 (+0.8% vs 4,299,866), diff
train.py-only (+26/-1).

### Surprises & Discoveries
- **SE is costly in this launch-bound regime even compiled** (planning smoke test): SE-k4 eager ~18ms/step (~2×
  plain k4's ~9ms), compiled reduce-overhead ~12.8ms → est ~60 epochs (vs plain compiled k4's 89). Compile only
  partially offsets SE's per-block kernel launches. This makes the experiment borderline: plain-k4 at ~60 epochs
  ≈ 95.6%, so SE must add ~+0.5pp to clear 96.10.
- Compile of the SE net succeeds cleanly (no graph breaks); compile cost ~20s (charged to budget).

### Decisions
- **SE every block, r=16** (standard SE-ResNet): kept the canonical placement/ratio rather than a reduced-count
  variant, to test evidenced SE. The epoch cost is accepted and flagged as a key informational metric.
- **Compile included as the enabler**: without it SE would run ~18ms → ~40 epochs (EXP-004-style starvation);
  compile keeps ~60. EXP-007 showed compiled-k4 alone = 95.92 (null accuracy effect), so SE remains attributable.

## Experimental Adjustments

(none yet)

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID; local background run)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08 (exit 0)

Description:
- Running the k=4 WideResNet + Cutout recipe with SE channel recalibration added to every block, compiled
  (reduce-overhead). Expect ~4,333,550 params, steady-state dt ~12–13ms, ~58–65 epochs, clean run < 600s. The
  test: does SE's channel gating add enough accuracy to clear 96.10 despite the ~17-epoch loss its cost imposes?
  A result near 96.0 would indicate SE has real per-epoch value but is offset by the epoch hit.

Observations:
- Clean startup, params 4,333,550 (run.log L1-2). No compile errors/graph breaks.
- **Throughput better than the smoke test predicted**: steady-state **dt = 9ms/step, ~13,900 img/s** (vs the
  planning smoke test's 12.8ms). The smoke test over-estimated SE cost — its per-step `torch.randn` data
  generation and shorter CUDA-graph warmup inflated the timing. Real sustained loop is faster. (run.log ep 3.)
- Implication: tracking toward **~80 epochs**, not ~60 — so SE gets a FAIR, well-trained test (not the feared
  epoch-starvation regime). Much better odds of a clean SE signal.
- Loss decreasing normally (1.24 by step 1150), no NaN.

Key Metrics:
- best_test_acc: **95.86%** @ best over 82 epochs — BELOW the 96.00 baseline and 96.10 bar (run.log summary).
- **num_epochs: 82** / num_steps 31,738 — a FAIR, well-trained test (≈ EXP-003's 77, EXP-007's 89). The feared
  epoch starvation did NOT happen (dt 9ms, not the smoke-test's 12.8ms). So SE got a clean shot.
- final_test_acc 95.85; final_test_loss **0.2083** (≈ EXP-007 0.2081, EXP-003 0.204 — SE did not reduce loss).
- num_params 4,333,550 (+0.8% vs 4,299,866 — SE added). peak_vram 455.7 MB.
- SE-k4 (95.86) ≈ compiled-k4 (95.92, EXP-007) within noise → **SE added no accuracy**.

## Verification Results

### Conditions Checked

- **Cond 1 — clean completion within budget**: PASS. best_test_acc 95.86% present, total_seconds 399.8 < 600, no
  traceback/recompile (run.log).
- **Cond 2 — metric ≥ 96.10**: **FAIL**. 95.86 < 96.10 (also < 96.00 baseline). → no-improvement.
- **Cond 3 — no constraint violations**: skipped — aborted after Cond 2 per protocol.

### Informational Metrics

- Recorded above. KEY point: num_epochs 82 means this is a clean (not epoch-starved) negative — SE genuinely
  did not help on this k=4 CIFAR model at this budget; it matched compiled-k4 within noise.

## Errors & Dead Ends

(none)

## Human Notes

> (none — autopilot)
