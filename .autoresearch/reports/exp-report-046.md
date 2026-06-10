# Report EXP-046: Clean +epoch control — off-budget compile-warmup with reduce-overhead kernels
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-046.md
- **Plan**: plans/plan-046.md
- **Log**: logs/exp-log-046.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) by editing only `train.py` within the fixed 300s training-compute budget on a single H20. Baseline = **96.22%** (EXP-012, commit 6c417a4); success bar = baseline + 0.1 = **96.32%**. The specific question this experiment tested: is the k=4 ResNet-20 + TrivialAugment recipe convergence-bound (would more epochs help?) or epoch-saturated at ~91 epochs — answered cleanly, free of the kernel-numerics confound that muddied EXP-045.

## Idea & Hypothesis
Chosen idea: apply ONLY the seed-safe compile-warmup from EXP-045 (one fwd+bwd on a zeros batch before the timed loop, `zero_grad`, reset BN buffers, no optimizer step) while KEEPING `mode="reduce-overhead"` — the project baseline's exact conv kernels. This reclaims the one-time compile cost from the per-step-timed 300s budget → net-new epochs at byte-identical numerics and recipe, isolating the pure epoch effect that EXP-045 entangled with max-autotune's ~0.3-0.5pp kernel-numerics penalty.

Hypothesis: realized epochs rise from ~91; IF the strongly-augmented TA recipe is convergence-bound, `best_test_acc` clears 96.32. The more likely outcome (given EXP-045's 96 ep → 95.71 and EXP-040's 94 ep → 95.91) is a landing within ±0.25pp of 96.22 — which, at baseline kernels, would close the saturation question kernel-independently and confirm EXP-045's regression was the kernel penalty.

## Approach
Single change to `train.py`: inserted the EXP-045-debugged warmup block immediately before `t_start_training` (L206), with L190 compile mode left UNCHANGED at `reduce-overhead`. The warmup runs in `model.train()`, does one bf16-autocast fwd+bwd on `torch.zeros(128,3,32,32, device=device).to(memory_format=torch.channels_last)` (the EXP-045 Run-2 fix — `memory_format` is NOT passed into the `torch.zeros` varargs-size call), then `optimizer.zero_grad(set_to_none=True)`, resets all `BatchNorm2d` running stats, and `torch.cuda.synchronize()`. No `optimizer.step()`, so weights stay at init and the timed loop begins byte-identical to baseline. The only deliberate difference from EXP-045 is the kernel mode. No recipe/hyperparameter/seed changes.

## Execution
One clean run on idle GPU 0 (shared 2-GPU H20 node; verified idle to avoid contention-confounded dt). Banner confirmed `ResNet-20 | params: 4,299,866`. Ran to completion in 408.7s wall (< 600 limit), no errors/NaN, peak_vram 455.3 MB. No retries — the warmup tensor construction was already debugged in EXP-045. Loss fell normally (ep1 45.70%, ep2 54.96%).

## Results
- **Primary metric**: best_test_acc 96.20% (baseline: 96.22, delta: −0.02, −0.02%) — within noise, below the 96.32 bar.
- **Observations**:
  - Warmup worked but reclaimed far less than hoped: startup_seconds 6.5 vs baseline ~2.1 → the reduce-overhead compile costs only **~4.4s** (NOT the ~14s the EXP-007 default-mode figure implied). So num_epochs rose only 91→**92** (+1), num_steps 35,797.
  - dt steady **8ms** (682/715 steps; 33×9ms) — the baseline reduce-overhead kernels, cleanly distinct from EXP-045's 7ms max-autotune kernels.
  - ep1 45.70% is NORMAL fast convergence vs EXP-045's anomalous ep1 26.4% → confirms EXP-045's slow start was a max-autotune kernel artifact, not a recipe property.
  - final_test_loss 0.1886 (< baseline ~0.195, run-to-run variation; +1 epoch lowered loss slightly but not top-1).
- **Analysis**: The hypothesis's "convergence-bound" branch is rejected and its "saturated" branch confirmed: +1 clean epoch at baseline numerics produced no top-1 change (96.20 ≈ 96.22). Critically, this **deconfounds EXP-045**: baseline-kernel +epochs lands at baseline acc (96.20) while EXP-045's max-autotune +epochs landed at 95.71 — the 0.49pp gap is therefore the max-autotune kernel-numerics penalty, NOT the epochs. The net is epoch-saturated at ~91, now established kernel-independently. A secondary finding: reduce-overhead compile is so cheap (~4.4s) that the off-budget-warmup avenue can never buy more than ~+1 epoch, so the throughput→buy-epochs strategy is fully closed (EXP-040/045/046).
- **Key Learning**: At baseline reduce-overhead kernels, +1 clean epoch = 96.20 (≈ baseline) → the k=4/300s/TA recipe is epoch-saturated at ~91 kernel-independently, and EXP-045's −0.51pp was purely the max-autotune kernel penalty.

## Verification
- **Conditions**: Condition 3 (`best_test_acc ≥ 96.32`) FAILED at 96.20; conditions 1 (clean run within budget), 2 (clean +epochs measurement delivered, though only +1), and 4 (no hard-constraint violations — `git diff --name-only` = `train.py` only; prepare.py/eval untouched; evaluate() once/epoch; no new deps; seed 42; warmup seed-safe) all PASSED.
- **Review Notes**: Results confirmed trustworthy. dt/startup/ep1 signatures all positively confirm baseline reduce-overhead kernels (not max-autotune), so the single-variable +epochs test was genuinely clean. No integrity concerns — warmup is seed-safe (zeros input/no step/BN reset) and does not touch eval.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid clean run, but the primary necessary condition failed — metric within noise of baseline, below the +0.1 bar.

## Unexplored Avenues
- **None for the epoch axis** — this experiment closes it. Combined with EXP-007 (default-mode +epochs, saturated) and EXP-045 (max-autotune +epochs, kernel-penalized), the recipe is epoch-saturated at ~91 across all three kernel regimes. No remaining variation of "give it more epochs within 300s" has an untested mechanism.
- **Normalization axis (Ghost BN)** — the one completely-untouched accuracy axis (brainstorm-046 candidate #2), deferred for its high reduce-overhead CUDA-graph dt-confound risk (per-block reshapes, EXP-042 gotcha). A confound-safe implementation (vmap-free, graph-stable BN reshape) would be needed to test it cleanly.
- **Stochastic depth (drop-path)** — untouched (candidate #3) but depth-driven regularizers have been null/regressive on this shallow 9-block net (dropout EXP-022, deep-supervision EXP-042, zero-γ EXP-026) and it carries the same CUDA-graph dt-confound; low prior.

## Next Steps
- **Document the 96.22 ceiling as fully mapped and pivot the next loop to a genuinely radical, dt-safe architectural change** (medium confidence) — capacity (×3), augmentation, LR, regularizer-adds, optimizer, bag-of-tricks, throughput→epochs are all closed. Per project-insights, compute/sequential-layer additions hit the epoch wall and compute-neutral changes hit the polish-vs-top1 wall.
- **Ghost BatchNorm with a CUDA-graph-stable implementation** (low confidence) — the only untouched accuracy axis; only worth it if a confound-free reshape is found (verify dt stays 8ms after the forward-signature change before trusting any accuracy delta).
- **Accept the plateau and stop spending loops on near-certain no-improvements** is NOT an option under the NEVER STOP directive — continue with radical/untested-axis attempts, thinking harder per the standing instruction.
