# EXP-018 Adversarial Plan Review

## Prioritized Concerns

### 1. The seven-snapshot gate is knife-edge

The `[88%,98%)` window spans 30 counted seconds. EXP-010's weak epochs took about 4.29 seconds, projecting endpoints near 88.6, 90.0, 91.4, 92.9, 94.3, 95.7, and 97.1%: exactly seven, with no margin. Charged snapshot time or a small weak-step slowdown can reduce this to six and abort production after almost five minutes. Add explicit margin or widen the window.

### 2. Controllers would test a reimplementation, not production logic

The plan described inline additions to monolithic `main()`, leaving no importable seam for arithmetic/refresh controllers. A copied controller implementation could pass while production diverges. Refactor snapshot, install/reset, and refresh behavior into importable `train.py` helpers and exercise those exact helpers.

### 3. Degenerate spread and hollow formal passes remain possible

`finite nonzero endpoint spread` has no meaningful floor and will pass nearly identical late iterates. Pre-register a minimum normalized spread. More importantly, `best_test_acc` includes online checkpoints, so an online checkpoint can clear 94.25 while the final SWA model contributes nothing; this could merge unsupported SWA code. Require the final SWA result itself to satisfy any improvement/attribution gate.

### 4. Default BN momentum is not a true recalculation

Resetting buffers and retaining default `momentum=0.1` produces an order-dependent EMA dominated by recent batches, not a one-pass cumulative recomputation. `num_batches_tracked` is then irrelevant to the update factor. Use the `torch.optim.swa_utils.update_bn` principle: temporarily set BN momentum to `None`, reset buffers/counters, accumulate, then restore original momenta.

### 5. No post-install optimizer step is not persisted

The planned provenance does not include the optimizer step at installation, so the final log cannot prove SGD stopped. Persist install-step and require it to equal final `num_steps`.

### 6. Step-retention margin is thin

Six refresh seconds plus a loose 1.5-second bookkeeping bound leaves only about 130 steps above the 26,091 floor. Evaluate snapshot count and step retention under the joint worst-case measured bounds, not separately.

### 7. Multi-pass refresh lifecycle is unspecified

Six seconds can traverse the 390-batch weak loader several times. Explicitly recreate iterators on exhaustion while reusing the persistent loader; otherwise refresh may stop after one pass.

## Provenance

- Reviewer: external Claude CLI, mandatory no-fallback path
- Command outcome: exit code 0
- Completed: 2026-08-06

