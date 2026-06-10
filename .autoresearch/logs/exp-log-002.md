# EXP-002: Cutout augmentation (16×16) on the k=4 WideResNet

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md
- **Plan**: plans/plan-002.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-002
- **Commit**: edf15d3 — on autoresearch/exp-002, merged to autoresearch/dev
- **PR**: not created — repo has no `origin` remote (intentional, TASK.md § Git Rules). Merged locally.
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented plan-002 in `train.py` only, on top of the EXP-001 k=4 WideResNet (integration-branch baseline).
Added `CUTOUT_SIZE=16`, a module-level `Cutout` callable (zeros one random 16×16 square on a normalized
CxHxW tensor using the seeded `torch.randint`), and appended `Cutout(CUTOUT_SIZE)` as the last element of
the training `transforms.Compose` (after ToTensor+Normalize). Everything else — model (k=4), recipe (bf16,
channels_last, cosine, Nesterov, label smoothing, WD 1e-4, PEAK_LR 0.2, batch 128), seed, and the frozen
eval transform — is unchanged. Syntax + ruff pass; Cutout sanity zeroed exactly 256 px (16×16).

### Surprises & Discoveries
- None. Cutout operates on the post-Normalize tensor, so zeroing sets the region to the dataset mean (std=1),
  which is the standard Cutout fill.

### Decisions
- Used `torch.randint` (not the `random` module) for hole coordinates so the augmentation randomness is
  covered by the existing `torch.manual_seed(42)` and remains deterministic — not seed hacking.
- Isolated change: Cutout only, WD kept at 1e-4, to attribute the regularization effect cleanly (WD bump is
  a separate follow-up if needed).

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash ID be24tf6vy (local, GPU 0)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: running
- **Started**: 2026-06-08
- **Ended**: pending

Description:
- Running the k=4 WideResNet with Cutout(16) added to training augmentation, under the fixed 300s budget on
  GPU 0, recipe otherwise unchanged. Tests the EXP-002 hypothesis that input-space regularization lifts
  best_test_acc above the 94.90% baseline (expected ~95%) by reducing overfitting of the high-capacity model.
  Watching that epoch count stays ~79 (Cutout is near-free) and loss stays healthy.

Observations:
- Started cleanly: `Device: cuda`, params 4,299,866 (unchanged from EXP-001), 390 batches/epoch (run.log L1-4).
- Throughput dt ~9-16ms/step — same as EXP-001; Cutout adds negligible cost. ~76 epochs projected.
- Loss ~1.03 @ ep5 (slightly higher than EXP-001 at the same point — expected, Cutout makes training harder),
  LR at peak 0.198, no NaN/divergence (run.log early step lines).

Key Metrics:
- best_test_acc: **95.42%** @ epoch 53 (baseline 94.90%, **+0.52 pp**) (source: run.log summary)
- final_test_acc: 95.25% | final_test_loss: **0.2169** (vs EXP-001 0.249 — less overfitting, as intended)
- training_seconds: 300.0 | total_seconds: 367.6 (source: run.log summary)
- peak_vram_mb: 490.8 (unchanged) | num_params: 4,299,866 (unchanged)
- num_epochs: **54** | num_steps: 21,052 (vs EXP-001 79 / 30,498 — ~31% FEWER epochs; the per-sample
  `torch.randint().item()` Cutout became a dataloader CPU bottleneck, dropping throughput ~9.8→~14ms/step)

## Verification Results

### Conditions Checked
- **Condition 1 — clean completion within budget**: PASS. `best_test_acc:` present, total_seconds 367.6
  < 600, 0 tracebacks (source: run.log summary; traceback grep=0).
- **Condition 2 — metric improvement (≥ 95.00)**: PASS. best_test_acc = **95.42%** ≥ 95.00; +0.52 pp over
  baseline 94.90 (source: run.log summary).
- **Condition 3 — no constraint violations**: PASS. `git diff --name-only autoresearch/dev` = only
  `train.py`; no pyproject/uv.lock diff; 54 eval lines = 54 epochs (eval once/epoch); seed unchanged (42).

All necessary conditions PASS → verified improvement.

### Informational Metrics
- final_test_loss 0.2169 (vs EXP-001 0.249) — Cutout reduced overfitting as hypothesized
- num_epochs/num_steps 54 / 21,052 (vs EXP-001 79 / 30,498) — Cutout impl cut throughput ~30%; a vectorized/
  cheaper Cutout could recover epochs and likely more accuracy (next-loop lead)
- peak_vram_mb 490.8 (unchanged)

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
