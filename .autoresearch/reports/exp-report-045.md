# Report EXP-045: Buy net-new epochs — compile-warmup off the timed budget + max-autotune
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-045.md
- **Plan**: plans/plan-045.md
- **Log**: logs/exp-log-045.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within a fixed 300s single-GPU training budget, editing only `train.py`. Baseline: **96.22%** (EXP-012, commit 6c417a4). Bar: **96.32%** (baseline + 0.1).

## Idea & Hypothesis
**Chosen idea**: After 34 consecutive no-improvements with every *accuracy* axis closed (capacity now closed all three ways incl. depth EXP-044), the binding constraint is the epoch wall and the one open question is whether the *TrivialAugment* recipe (added EXP-012, after EXP-007's pre-TA saturation finding) is convergence-bound or epoch-saturated at ~91 epochs. No experiment had ever genuinely *added* net epochs. This experiment does, via two changes serving one goal: (1) a seed-safe **compile-warmup** before the timed loop, moving the one-time torch.compile cost off the per-step-timed 300s budget (the timer at L218 starts after the dataloader yields, so the step-1 compile was being billed to the budget); (2) **max-autotune** compile mode, the one untried dt reducer (EXP-040).

**Hypothesis**: epochs rise from ~91 to ≥100; IF the TA recipe is convergence-bound, the extra epochs lift best_test_acc ≥0.1pp (≥96.32); IF epochs rise but accuracy stays flat, the net is epoch-SATURATED — closing the last open question.

## Approach
Two `train.py` changes, recipe otherwise byte-identical. (1) L192 `mode="reduce-overhead"` → `mode="max-autotune"`. (2) Inserted a warmup before `t_start_training`: one fwd+bwd on a `torch.zeros(128,3,32,32)` channels_last batch through `compiled_model` under bf16 autocast, then `optimizer.zero_grad(set_to_none=True)`, reset all BatchNorm2d running stats, `torch.cuda.synchronize()` — no `optimizer.step()`. Seed-safety: zeros input consumes no RNG, no dropout in the model, weights never stepped (stay at kaiming init), BN buffers reset to pristine → the training loop begins byte-identical to baseline, so any accuracy delta is attributable to epoch count. Ran on idle GPU 1.

## Execution
Run 1 crashed instantly at the warmup: `torch.zeros(...)` rejects `memory_format` in its varargs-size signature. Fixed (build then `.to(memory_format=channels_last)`, mirroring the real-input conversion) — one retry. Run 2 completed cleanly in 510.3s total wall. **The mechanism worked exactly as designed**: `startup_seconds` rose 2.1→**79.1** (the ~77s max-autotune compile moved entirely off the per-step budget into startup), max-autotune cut **317/746 sampled steps to 7ms** (rest 8ms; below the reduce-overhead 8ms floor), and `num_epochs` rose **91→96** / 37,342 steps. No errors/NaN; converged to a flat tail (ep94 95.67, ep95 95.71, ep96 95.70).

## Results
- **Primary metric**: best_test_acc **95.71%** (baseline 96.22, delta **−0.51pp**, −0.53%). final_test_loss 0.2054 (> 0.195). num_epochs 96, startup 79.1s, dt mean ~7.6ms, peak_vram 489.7MB.
- **Observations**: The experiment cleanly delivered its mechanism — for the first time, net epochs were genuinely ADDED (91→96) at the byte-identical recipe (the warmup off-budget trick + max-autotune's sub-8ms steps), and `startup_seconds` 79.1 confirms the compile cost moved off the budget. Yet best_test_acc FELL to 95.71. Critically, this mirrors EXP-040 (cudnn.benchmark: 94 epochs → 95.91, −0.31pp): **both throughput-variant runs land ~0.3-0.5pp below baseline despite ≥ baseline epochs.** The common factor is that both swapped in throughput-optimal conv kernels (max-autotune Triton templates / cuDNN-benchmark algos) different from the reduce-overhead baseline's — strongly suggesting those kernels' numerics cost a small but real ~0.3-0.5pp.
- **Analysis**: Two effects are entangled. (a) **Saturation**: +5 epochs produced no gain (acc flat-to-down, converged tail) — consistent with the TA recipe being epoch-saturated at 91, reinforcing EXP-007's pre-TA finding and EXP-040's 94-epoch null. (b) **Kernel-numerics confound**: the −0.51pp regression exceeds the ±0.25pp jitter band and matches EXP-040's sign/magnitude, so max-autotune's kernel change likely contributes a real penalty, meaning this is NOT a perfectly clean isolation of the epoch effect. Either way the headline holds: buying ~5 epochs via faster kernels does not break the plateau, and the throughput→epochs avenue (the last lever flagged by EXP-040) does not yield top-1 gains. The 96.22 plateau is now confirmed against the final open lever.
- **Key Learning**: Genuinely adding ~5 net epochs (warmup off-budget + max-autotune 7ms steps, 91→96 ep) did NOT raise top-1 (95.71, −0.51pp, converged) → the TrivialAugment recipe is epoch-saturated at ~91; and throughput-optimal kernels (max-autotune/cudnn-benchmark) cost ~0.3-0.5pp vs the reduce-overhead baseline (mirrors EXP-040).

## Verification
- **Conditions**: condition 3 (best_test_acc ≥ 96.32) FAILED (95.71). Conditions 1 (clean completion, 510s < 600) and 4 (scope: only train.py; seed-safe warmup) PASSED. Condition 2 (epoch-count signal) DELIVERED (91→96, startup 79.1 confirms off-budget compile).
- **Review Notes**: Results trustworthy as a no-improvement. The run is valid, on-scope, seed-safe (warmup uses zeros/no-step/BN-reset → comparable run), evaluated once/epoch on the frozen harness. The max-autotune kernel-numerics effect is a confound on the *clean* saturation question (not an integrity failure) — flagged, not disqualifying. No false pass/fail.
- **Verdict**: no-improvement
- **Verdict Basis**: primary necessary condition failed (95.71 < 96.32); valid, clean, on-scope run (not invalid/crash).

## Unexplored Avenues
- **Clean saturation control: compile-warmup + reduce-overhead (NO max-autotune)** — adds the same ~+5 epochs but with the BASELINE's exact conv kernels, isolating the pure epoch effect from the max-autotune numerics penalty. If that also lands flat/below baseline, saturation is confirmed kernel-independently. This is the natural EXP-046 (medium confidence it confirms saturation cleanly).
- **Larger epoch increase** — to decisively test saturation one would want +20-30 epochs, but the dt floor (now ~7ms with max-autotune) caps the achievable increase; off-the-timed-path augmentation (moving GPU Cutout into CPU dataloader workers) was rejected earlier (EXP-002 dataloader bottleneck). No clean path to a big epoch jump exists at this budget.
- **The kernel-numerics finding itself** — if throughput-optimal kernels reliably cost ~0.3-0.5pp, then the reduce-overhead baseline's kernels are near-accuracy-optimal; not an avenue for gain, but confirms the recipe is well-chosen.

## Next Steps
- **EXP-046: compile-warmup + reduce-overhead clean control (confidence: medium it confirms saturation)** — isolate the +5-epoch effect from max-autotune numerics; the cleanest remaining probe of the one open question. If null, the saturation question closes kernel-independently and the throughput→epochs avenue is fully exhausted.
- **Document 96.22 as the fully-mapped k=4/300s ceiling (confidence: high)** — every accuracy axis (capacity all-3-ways, aug, schedule, regularizers, optimizer family, objective, head, feature-routing, normalization-untested-but-low-EV) AND the throughput→epochs lever are now closed; 35 straight no-improvements.
- **Only genuinely-untried axis left: normalization (Ghost BN) — confidence: low** — a fresh axis but uncertain benefit at batch 128 and real torch.compile dt-confound risk (EXP-042 graph-break gotcha); deprioritized below the clean control.

## Exit Action Results
- None defined for this goal — skipped.
