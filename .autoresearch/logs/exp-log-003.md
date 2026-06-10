# EXP-003: Vectorized GPU Cutout (recover throughput)

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md
- **Plan**: plans/plan-003.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-003
- **Commit**: f59de56 — on autoresearch/exp-003, merged to autoresearch/dev
- **PR**: not created — repo has no `origin` remote (intentional, TASK.md § Git Rules). Merged locally.
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented plan-003 in `train.py` only. Replaced the per-sample CPU `Cutout` class (and its `train_tf`
entry) with a module-level `cutout_batch(x, size)` that builds a `(B,H,W)` boolean hole mask via broadcast
`arange` comparisons and `masked_fill`s the batch to 0 — fully vectorized, no `.item()` host-sync. Applied it
in the training loop right after the inputs are moved to device (channels_last), before the autocast forward.
Cutout semantics are unchanged (one random ≤16×16 hole per image, border-clipped, zeroed in normalized space).
Everything else fixed. Syntax + ruff pass; sanity showed independent per-image holes (160/256/256/176 px,
border clipping visible) with all other pixels unchanged.

### Surprises & Discoveries
- None. The arange-comparison mask naturally reproduces EXP-002's `max(0,·)/min(h,·)` border clipping
  (holes near edges are smaller), so regularization strength matches EXP-002.

### Decisions
- Applied Cutout in the training loop (per batch) rather than in the dataset transform — this is what removes
  the dataloader CPU bottleneck. Uses cuda `torch.randint` (seeded by `torch.cuda.manual_seed(42)`): deterministic,
  not seed hacking.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash (filled at launch), local GPU 0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08

Description:
- Running the k=4 WideResNet + Cutout(16) with Cutout moved to a vectorized GPU batch op, under the fixed 300s
  budget on GPU 0. Tests the EXP-003 hypothesis that removing the EXP-002 dataloader bottleneck restores
  throughput to ~79 epochs and lifts best_test_acc above 95.42% via more training of the regularized model.
  KEY signal to watch: num_epochs should recover toward ~75–79 (vs EXP-002's 54).

Observations:
- Throughput recovered: at 10.6% budget reached step 3100/ep 8 vs EXP-002's step 1950/ep 5 — GPU Cutout is
  near-free (dt ~10ms). Loss healthy ~0.95 @ ep8, no NaN (run.log early lines).

Key Metrics:
- best_test_acc: **96.00%** @ epoch 73 (baseline 95.42%, **+0.58 pp**) (source: run.log summary)
- final_test_acc: 95.85% | final_test_loss: **0.2044** (vs EXP-002 0.217 — even less overfit with more epochs)
- num_epochs: **77** | num_steps: 29,931 (vs EXP-002 54 / 21,052 — throughput restored to ~EXP-001 level)
- training_seconds: 300.0 | total_seconds: 384.8 | peak_vram_mb: 492.1 | num_params: 4,299,866

## Verification Results

### Conditions Checked
- **Condition 1 — clean completion within budget**: PASS. `best_test_acc:` present, total_seconds 384.8 < 600,
  0 tracebacks (run.log summary; traceback grep=0).
- **Condition 2 — metric improvement (≥ 95.52)**: PASS. best_test_acc = **96.00%** ≥ 95.52; +0.58 pp over 95.42.
- **Condition 3 — no constraint violations**: PASS. only `train.py` changed; no pyproject/uv.lock diff; 77 eval
  lines = 77 epochs (eval once/epoch); seed unchanged (42).

All necessary conditions PASS → verified improvement.

### Informational Metrics
- num_epochs/num_steps: 77 / 29,931 (vs EXP-002 54 / 21,052) — confirms the dataloader bottleneck is fixed
- final_test_loss 0.2044 (vs 0.217) — slightly less overfit with the extra epochs
- peak_vram_mb 492.1 (unchanged)

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
