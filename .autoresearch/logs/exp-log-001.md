# EXP-001: Widen the ResNet (WideResNet-style, k=4) + projection shortcuts

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-001.md
- **Plan**: plans/plan-001.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-001
- **Commit**: 0086a21 — on autoresearch/exp-001, merged to autoresearch/dev
- **PR**: not created — repo has no `origin` remote (intentional, TASK.md § Git Rules). Merged locally.
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented plan-001's architecture changes in `train.py` only, on top of the EXP-000 recipe (which is the
integration-branch baseline). Three edits: (1) `BasicBlock` now uses a 1×1-conv + BN projection shortcut on
downsample/channel-change blocks (`self.shortcut`), replacing the channel-padding identity; identity blocks
use `nn.Identity()`. (2) `ResNet.__init__` takes `width_mult=k`: stem stays 16 channels, stages become
{16k, 32k, 64k}, fc input 64k — k=1 reproduces the original net. (3) Added `WIDTH_MULT=4` and pass it at
construction. All recipe knobs unchanged. Syntax + ruff pass. Param sanity: k=4 → 4,299,866 params (15.8×
the 272k k=1 net), forward returns (N,10).

### Surprises & Discoveries
- k=1 yields 272,474 params vs the original 269,722 — the small difference is the new 1×1 projection convs +
  BN on the two downsample blocks (the old channel-pad was parameter-free). Expected and immaterial.
- `_weights_init` (kaiming_normal on Conv2d/Linear) automatically covers the new 1×1 shortcut convs; BN layers
  keep their default init (weight=1, bias=0), as intended.

### Decisions
- **k=4 ({64,128,256})** chosen as the primary capacity jump (WRN-style strong CIFAR width; VRAM free). Risk:
  more FLOPs → fewer epochs in 300s. The time-fraction cosine anneals fully regardless of step count, so the
  main risk is underfit if epochs are very low — flagged for the analysis phase. Fallback k=2 noted in brainstorm.
- **Recipe held fixed** (PEAK_LR 0.2, WD 1e-4, label smoothing 0.1, batch 128) to isolate the architecture
  effect; wider nets may prefer different LR/WD — deferred to a follow-up.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash ID bpl3s0gok (local, GPU 0)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: running
- **Started**: 2026-06-08
- **Ended**: pending

Description:
- Running the widened (k=4, 4.3M-param) ResNet with projection shortcuts under the fixed 300s budget on GPU 0,
  EXP-000 recipe otherwise unchanged. Tests the EXP-001 hypothesis that added capacity lifts best_test_acc
  meaningfully above the 92.06% baseline (target ~93%+), since EXP-000 showed the model was capacity-bound.
  Watching that enough epochs fit in 300s to converge (underfit is the main risk).

Observations:
- Started cleanly: `Device: cuda`, params **4,299,866** (15.8× baseline), 390 batches/epoch (run.log L1-4).
- Throughput: dt ~9-18ms/step vs baseline ~6-7ms — only ~1.5-2.5× slower despite 15.8× params, confirming
  the small model was overhead-bound and the H20 absorbs the extra FLOPs. ~99 steps/s → projecting ~75
  epochs in 300s, so underfit risk is low (run.log early step lines).
- Loss healthy (~0.82 label-smoothed @ ep7), LR at peak 0.199 (warmup complete), no NaN/divergence.

Key Metrics:
- best_test_acc: **94.90%** @ epoch 75 (baseline 92.06%, **+2.84 pp**) (source: run.log summary)
- final_test_acc: 94.83% | final_test_loss: 0.2491 (source: run.log summary)
- training_seconds: 300.0 | total_seconds: 385.7 | startup_seconds: 1.2 (source: run.log summary)
- peak_vram_mb: 490.8 (vs EXP-000 164 — still tiny vs 98 GB) (source: run.log summary)
- num_epochs: 79 | num_steps: 30,498 (vs EXP-000 109 / 42,156 — ~28% fewer epochs for 15.8× capacity;
  worth it) (source: run.log summary)
- num_params: 4,299,866 (15.8× the 272k k=1 net) (source: run.log summary)

## Verification Results

### Conditions Checked
- **Condition 1 — clean completion within budget**: PASS. `best_test_acc:` present, total_seconds 385.7
  < 600, 0 tracebacks (source: run.log summary; traceback grep=0).
- **Condition 2 — metric improvement (≥ 92.16)**: PASS. best_test_acc = **94.90%** ≥ 92.16; +2.84 pp over
  baseline 92.06 (source: run.log summary).
- **Condition 3 — no constraint violations**: PASS. `git diff --name-only autoresearch/dev` = only
  `train.py`; no pyproject/uv.lock diff; 79 eval lines = 79 epochs (eval once/epoch); seed unchanged (42).

All necessary conditions PASS → verified improvement.

### Informational Metrics
- peak_vram_mb: 490.8 (15.8× model still uses <0.5% of 98 GB — width remains essentially free)
- num_epochs / num_steps: 79 / 30,498 (vs EXP-000 109 / 42,156 — capacity bought a +2.84 pp jump at the
  cost of ~28% fewer epochs; clearly worth it)
- num_params: 4,299,866 (confirms ~16× capacity increase)

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
