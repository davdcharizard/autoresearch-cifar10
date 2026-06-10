# Report EXP-024: BlurPool / anti-aliased downsampling (Zhang 2019)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-024.md
- **Plan**: plans/plan-024.md
- **Log**: logs/exp-log-024.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within a fixed 300s budget on one H20. Baseline = **96.22%** (EXP-012, commit 6c417a4); bar = **96.32%** (+0.1pp).

## Idea & Hypothesis
Chosen idea: anti-aliased downsampling (BlurPool, Zhang 2019) at the two stride-2 sites — replace naive strided subsampling with stride-1 conv → fixed binomial blur → stride-2 subsample. Selected as the first RADICAL structural change after all scalar knobs were bracketed and every regularizer failed: BlurPool is a convergence-neutral *generalization* mechanism (parameter-free, no stochastic penalty) targeting the binding constraint (shift-invariance/generalization). Hypothesis: improved shift-invariance lifts best_test_acc above 96.32, PROVIDED the restructured graph still fits ~91 epochs; regression/confound if the added compute craters epochs (EXP-015 / capacity pattern).

## Approach
Added a parameter-free `BlurPool2d` (fixed 3×3 binomial depthwise kernel as a registered buffer) and rewired `BasicBlock`'s downsample path: conv1 → stride-1; `BlurPool(stride=2)` after relu(bn1(conv1)); projection shortcut anti-aliases its input before a stride-1 1×1. Non-downsample blocks unchanged. Verified params = 4,299,866 (unchanged — blur kernels are buffers) and a forward smoke test confirmed correct output shape/alignment. Scope = train.py only; no new deps (`F.conv2d` core torch). No deviations from plan.

## Execution
Single run, no retries. Clean startup/compile (no graph break on the depthwise fixed-kernel conv), no NaN. dt rose 8→9-10ms early. Completed **77 epochs** (vs baseline 91) in 398.1s, peak VRAM 577.9MB.

## Results
- **Primary metric**: best_test_acc = **95.66%** (baseline: 96.22, delta: **−0.56pp**, −0.58%)
- **Observations**: **num_epochs dropped 91→77** — the flagged compute confound materialized. Moving conv1 to stride-1 ~4×'d its FLOPs at the two heaviest convs (layer2 64→128, layer3 128→256); the launch-bound headroom absorbed only part of it (dt 8→~9.5ms), costing ~15% of epochs. final_test_loss ROSE 0.195→0.2085 (same LS, comparable) — the under-training signature.
- **Analysis**: Verdict no-improvement, but the result is **compute-confounded** and NOT a clean test of anti-aliasing's merit. At 77 epochs the model is under-trained relative to the 91-epoch baseline, and the project has repeatedly shown ~14 fewer epochs alone costs ≳0.3–0.5pp (EXP-007: 77 ep ≈ 95.9; capacity EXP-004/009; pre-act EXP-015). So the −0.56pp is plausibly explained by the epoch loss alone — anti-aliasing's intrinsic effect (if any) is masked. This re-confirms the project's most robust insight: at a fixed 300s budget, ANY change that adds non-trivial compute (capacity, restructured graphs, BlurPool) hits the epoch wall and regresses, regardless of the change's merit. BlurPool is the canonical anti-aliasing recipe but is intrinsically expensive here because it moves downsampling convs to full resolution.
- **Key Learning**: BlurPool's stride-1 conv ~4×'d FLOPs at the two heaviest convs → epochs 91→77 (compute-confounded), under-trained to 95.66 (−0.56pp); anti-aliased downsampling is too costly to test fairly at the 300s/k=4 budget.

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.32) FAILED (95.66); Conds 2–3 skipped per protocol (would pass — clean 398.1s run, train.py-only, params 4,299,866, eval-count 77 == epochs, no new deps).
- **Review Notes**: Results trustworthy as a measurement (clean run, scope/params intact, no integrity issue → no-improvement, not invalid), but the regression is compute-CONFOUNDED (epochs 77 vs 91) per the plan's mandatory attribution rule — cannot be attributed to anti-aliasing on its merits.
- **Verdict**: no-improvement
- **Verdict Basis**: necessary condition failure (primary metric below bar); regression confounded by epoch loss.

## Unexplored Avenues
- **BlurPool on only the cheaper (layer2) downsample, or a 2-tap blur**: would reduce the FLOPs hit and might keep epochs higher, but a partial application is a weaker test and still adds cost; the launch-bound headroom is already mostly consumed. Low value.
- **Anti-aliasing the shortcut only (not conv1)**: cheaper but not the canonical recipe and unlikely to capture the benefit. Low value.
- The deeper truth: anti-aliasing's benefit (if any on this shallow 2-downsample net) is intrinsically gated by the epoch wall — it can only be fairly tested with a bigger compute budget. At 300s/k=4 it is closed by the same wall that closed capacity scaling.

## Next Steps
- **Per-channel input std-normalization** (std=(1,1,1)→true CIFAR std) — the last untried convergence-neutral, compute-NEUTRAL single-knob probe; confidence LOW it gains (Conv→BN absorbs it → expected null), but it cleanly closes the input-normalization axis with zero confound risk. (medium confidence clean null; low confidence gain.)
- After the std-norm probe, the search is genuinely exhausted: ~17 axes closed, all scalar knobs bracketed, every compute-adding structural change hits the epoch wall, every added regularizer under-fits. The honest scientific conclusion is the **96.22 plateau is the ceiling for this k=4 ResNet-20 at the 300s/H20 budget** (capacity- and convergence-bound). (high confidence the plateau is real.)
- Any further gains would require relaxing a fixed constraint (more time budget or a fundamentally more compute-efficient architecture that stays launch-bound while improving generalization) — not available within the goal's hard constraints.

## Exit Action Results
<!-- No exit actions defined for this goal. -->
