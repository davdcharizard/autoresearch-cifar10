# Report EXP-008: Squeeze-Excitation blocks on k=4 (+ torch.compile enabler)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-008.md
- **Plan**: plans/plan-008.md
- **Log**: logs/exp-log-008.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%) under a fixed 300s budget on one H20, editing only `train.py`. Higher is
better. Baseline **96.00%** (EXP-003); success bar ≥ **96.10%**.

## Idea & Hypothesis
Chosen idea: add Squeeze-Excitation (SE) channel recalibration to every k=4 BasicBlock — the first test of an
*accuracy-per-epoch architectural* lever, orthogonal to the four saturated axes (width, regularization, weight-
averaging, training-length). `torch.compile(reduce-overhead)` included as the *enabler* to offset SE's per-block
launch cost (which would otherwise epoch-starve the run, the EXP-002/EXP-004 trap). Hypothesis: SE's learned
channel gating lifts `best_test_acc` past 96.10 (expected ~96.2–96.6%) at a viable epoch count.

## Approach
`train.py`-only edits: added `SEModule` (GAP → FC(C→C/16) → ReLU → FC → sigmoid → per-channel scale), applied in
`BasicBlock.forward` to the post-bn2 residual branch before the add; `SE_REDUCTION=16`; compiled the training
model (reduce-overhead) and trained through it; eval on the eager `model`. +0.8% params (4,299,866→4,333,550).
All else fixed (k=4, Cutout, PEAK_LR 0.2, WD 1e-4, label smoothing, batch 128, bf16, channels_last, Nesterov,
cosine, seed 42).

## Execution
One run, no retries/errors, clean compile (no graph breaks). **Throughput beat the planning smoke test**:
steady-state dt = 9ms/step (~13,900 img/s), not the predicted 12.8ms — the smoke test's per-step `torch.randn`
data generation and short CUDA-graph warmup had inflated its estimate. Consequently the run fit **82 epochs**
(not the feared ~60), i.e. a *fair, well-trained* test comparable to EXP-003 (77) and EXP-007 (89). Completed in
399.8s total, peak VRAM 455.7 MB.

## Results
- **Primary metric**: **95.86%** (baseline 96.00%, delta **−0.14 pp**, −0.15%) — below the +0.1 bar and baseline.
- **Observations**: SE-k4 (95.86) ≈ compiled-k4 (95.92, EXP-007) — within the ~0.2pp noise band, i.e. **SE added
  no measurable accuracy**. final_test_loss 0.2083 ≈ EXP-007 (0.2081) ≈ EXP-003 (0.204): SE did not reduce the
  test loss either. Late-epoch evals plateaued at 95.78–95.86 (ep 77–82).
- **Analysis**: This is a *clean* negative, not a confounded one — the feared epoch starvation did not occur (82
  epochs, dt 9ms), so SE got a fair shot and still did nothing. Channel-attention recalibration does not help
  this wide ResNet-20 on CIFAR-10 at this budget; the network is evidently not bottlenecked by lack of adaptive
  channel gating. Combined with prior loops, this exhausts a **fifth** lever: width (≤k4), regularization,
  weight-averaging, training-length, and now architectural channel-attention are all at their local ceiling.
  96.0% is a robust plateau for this model/recipe/budget.
- **Key Learning**: SE channel recalibration adds no accuracy to the k=4 WideResNet on CIFAR-10 at this budget
  even with a fair (82-epoch) training run (95.86 ≈ compiled-k4 95.92) — the model is not channel-gating-limited.

## Verification
- **Conditions**: Cond 1 (clean completion in budget) PASS; Cond 2 (≥96.10) **FAIL** (95.86); Cond 3 skipped.
- **Review Notes**: Trustworthy — clean single run, frozen eval, seed 42, one eval/epoch, num_params confirms SE
  was added (+0.8%), compile is execution-only and EXP-007 showed it has null standalone accuracy effect → the
  null result is attributable to SE itself, not to under-training (82 epochs) or compile. No reward-hacking
  surface. The −0.14pp is within noise.
- **Verdict**: no-improvement
- **Verdict Basis**: valid, trustworthy, well-trained run; primary metric below the +0.1 bar (cond 2 failed).

## Unexplored Avenues
- **Cheaper channel attention (ECA)**: replaces SE's 2 FCs with a 1×k 1D conv — but SE already got a fair
  82-epoch test and showed no signal, so a near-identical mechanism is unlikely to differ. Low priority.
- **Spatial/hybrid attention (CBAM)**: adds a spatial-attention map (different mechanism than channel gating).
  Heavier (more kernels), and given channel attention's null result, expected low. Low priority.
- **Compiled k=5 capacity (the still-open lever)**: EXP-007 established compile re-opens capacity; we have NOT
  tested k=5. Pure capacity is the one historically-dominant axis not yet cleanly retried with compile —
  higher-upside than more attention variants, though epoch-bound (~60 ep).
- **Better stem / activation (SiLU), or 2-conv stem**: changes the feature front-end rather than per-block gating.

## Next Steps
1. **Compiled k=5 ({80,160,320}, ~6.7M)** — *medium confidence*; the one historically-dominant lever (capacity)
   not yet cleanly tested with the compile enabler. Risk: ~60 epochs (width eats compile savings) → possible
   mild under-training, but it's the highest-upside remaining axis. *Best next experiment.*
2. **Mixup/CutMix augmentation** — *low-medium confidence*; a genuinely different regularizer (interpolation,
   not occlusion like Cutout) — but typically needs more epochs to pay off, a poor fit for the 300s budget.
3. **Accept 96.0% as a hard plateau** — *strategic*; FIVE axes now saturated. Remaining moves are marginal/
   noise-scale; capacity (k=5) is the last clearly-untried high-upside lever before declaring convergence.

## Exit Action Results
- None defined for this goal — skipped.
