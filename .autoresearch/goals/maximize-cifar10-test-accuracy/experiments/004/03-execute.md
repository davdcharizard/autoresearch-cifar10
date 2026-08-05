# EXP-004: Identity-initialized (ReZero) layer2 residual block — capacity probe

## Execution

Overall Status & Info:
- **Created**: 2026-06-28
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-004
- **Commit**: ae31206 (on autoresearch/maximize-cifar10-test-accuracy-004; merged/fast-forwarded to integration branch)
- **PR**: N/A — no git remote configured (local-only per TASK.md)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the EXP-004 capacity probe additively on the EXP-003 base (DavidNet + whitening + EMA + flip-TTA, 95.87%). Two edits to `train.py`: (1) added a `GatedResidual(nn.Module)` class next to `Residual` — `c1=conv_bn(c,c)`, `c2=conv_bn(c,c)`, and a learnable scalar `self.alpha = nn.Parameter(torch.zeros(1))`, with `forward: x + alpha * c2(c1(x))` (ReZero, Bachlechner et al. 2020); (2) appended `GatedResidual(256)` to `self.layer2` (line 129), which previously had no residual block. `PEAK_LR` and all other HPs unchanged (single-variable capacity test at the validated LR). Milestone 1 passed: `py_compile` clean; in-process smoke confirmed (a) identity at init (`allclose(block(h),h)`, alpha=0), **(b) the gate receives a nonzero gradient (alpha.grad=0.0179 — block is trainable, NOT dead)**, (c) full forward shape `[2,10]` finite, (d) pool input `512×4×4` (spatial chain intact), (e) learnable params 7,783,169 (exact), (f) whiten frozen. Total params 7,784,627.

### Surprises & Discoveries
- The plan-phase adversarial review caught a **fatal bug** in the originally-planned identity-init: zeroing the new block's final BatchNorm γ would make `c2(c1(x))=ReLU(0)=0` with ReLU'(0)=0, so **no gradient would ever reach the block** (it stays identity forever — testing "same net, fewer epochs", not capacity). The fix (ReZero scalar gate) keeps a live gradient path. The smoke's gradient check (alpha.grad=0.0179≠0) directly confirms the bug is avoided — this was the single most important pre-run check.

### Decisions
- **ReZero gate instead of BN-zero identity init** — the only correct way to realize "start identity, earn capacity gradually" given `conv_bn`'s post-BN ReLU. Documented in `02-plan.md` Design correction.
- **PEAK_LR held at 0.4** — the gate's gradual capacity ramp removes the stability rationale for the airbench 0.78× LR cut, giving a clean single-variable capacity A/B. `alpha` is in the SGD param group (gets WD 5e-4 — negligible on a gradient-driven scalar).
- **Only layer2 gains the gated block** — `Residual` (layer1/layer3) is untouched, keeping the proven blocks' kaiming init for cleanest attribution.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (local, background) — PID recorded at launch
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-28
- **Ended**: 2026-06-28 (exit 0)

Description:
- Official EXP-004 run: `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` on GPU 1. Adds a ReZero-gated `Residual(256)` block to layer2 (the only stage without one) on the EXP-003 recipe (whitening+EMA+flip-TTA, otherwise byte-identical), to test whether added representational capacity lifts `best_test_acc` from 95.87% within the same 300s training budget. Expected ~95.95–96.1% (central ~96.0%), bar ≥95.97%. Watch: early img/s → projected epochs ≥~130; ep1/ep10 test_acc tracking EXP-003 (60.19%/85.45%) since the gate starts as identity; alpha ramping off zero; no divergence; wall < 600s.

Observations:
- **Cleared the bar: best_test_acc 96.00%** (baseline 95.87%, +0.13pp), first reached at ep119 and held through ep142 (final 96.00%). No divergence, no NaN. (source: run.log summary + `eval ep` trace)
- **Identity-gate start preserved the early trajectory** (mechanism working as designed): ep1 58.70% / ep10 85.19% — within noise of EXP-003's ep1 60.19% / ep10 85.45%, i.e. the ReZero block did NOT disrupt early convergence (it starts as exact identity). (source: run.log eval ep 1/10 vs experiments/003 trace)
- **Capacity advantage emerged mid-training**: ep25 92.63% vs EXP-003's 88.84% (**+3.79pp**), ep50 94.00% — the added block earned its capacity as `alpha` ramped off zero, lifting the mid/tail floor. (source: run.log eval ep 25/50)
- **142 epochs / 13,704 steps** in 300.0s training (vs EXP-003's 174 ep) — 32 fewer epochs from the ~11% per-step throughput cost of the extra 8×8 block (img/s ~26.3k steady vs EXP-003 ~29.3k); the capacity gain more than paid for the lost epochs. (source: run.log summary + step lines)
- Wall 445.2s < 600s cap; peak VRAM 1635 MB (≈ EXP-003's 1614 — the 8×8 block's activations are tiny); whitening 0.46s off-budget. (source: run.log summary)

Key Metrics:
- best_test_acc: 96.00% @ ep119 (held to ep142) (source: run.log "best_test_acc:    96.00%")
- final_test_acc: 96.00% @ ep142 | final_test_loss: 0.3247 (source: run.log summary)
- training_seconds: 300.0 | total_seconds: 445.2 | startup_seconds: 1.9 | whitening_seconds: 0.46 (source: run.log summary)
- num_epochs: 142 | num_steps: 13704 | peak_vram_mb: 1635.4 | num_params: 7,784,627 (source: run.log summary)
- Early trajectory: ep1 58.70% / ep10 85.19% / ep25 92.63% / ep50 94.00% (source: run.log eval trace)

## Verification Results

### Conditions Checked

1. **Clean run within wall guard** — PASS. Background process exited 0; `grep -c "^best_test_acc:" run.log` == 1; `total_seconds 445.2` < 600 (not timeout-killed); full summary present. (source: run.log)
2. **Full training budget + scope intact** — PASS. `training_seconds 300.0` ≥ 295. `git diff --quiet -- prepare.py` and `git diff --quiet <dev> -- prepare.py` both exit 0 (prepare.py byte-unchanged). `git diff --name-only <dev>` lists only `train.py`. **Diff-content:** the diff is limited to (i) the new `GatedResidual` class and (ii) the one-token `layer2` change (22 insertions, 1 deletion); `_forward_once`, `forward` (TTA), `compute_whitening_weight`, the eval call, the training loop, and all HP constants are untouched. (source: git diff)
3. **Improvement ≥ +0.1pp + genuineness** — PASS → **improvement**. `best_test_acc 96.00%` ≥ bar 95.97% (baseline 95.87%, **+0.13pp**). **Genuineness:** max per-epoch `best:` across the trace = 96.00% = summary (and max per-epoch `test_acc:` = 96.00%, from real `Eval.evaluate` readings — not fabricated). **Reward-hack/leakage:** exactly one `evaluator.evaluate(` call site (once per epoch); seeds unchanged — `torch.manual_seed(42)` + `torch.cuda.manual_seed(42)` + the local whitening `Generator().manual_seed(0)` (3 legitimate pre-existing seeds, no new seed line, no seed search); no `train=False`/`test_set`/`testset` leakage; only `datasets.CIFAR10(train=True)`; `forward()` untouched (still flip-TTA only, per the diff). (source: run.log; train.py; git diff)

**All necessary conditions passed → verdict: improvement.**

### Informational Metrics
- peak_vram_mb: 1635.4 (source: run.log summary) — ≈ EXP-003's 1614; the frozen-budget + tiny 8×8 block keep VRAM flat (non-binding vs 98 GB).
- training_seconds: 300.0 | num_epochs: 142 | num_steps: 13704 (source: run.log) — 32 fewer epochs than EXP-003's 174 (the capacity/throughput trade), yet higher best_acc.
- total_seconds: 445.2 (source: run.log) — wall vs 600s cap (≈ EXP-003's 452.8s).
- num_params: 7,784,627 total / 7,783,169 learnable (source: run.log + smoke) — +1,180,673 over EXP-003 (two `conv_bn(256,256)` + the `α` scalar).

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
