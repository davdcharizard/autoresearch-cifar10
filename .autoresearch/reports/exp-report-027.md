# Report EXP-027: ResNet-D downsample (avgpool-2 + 1×1-stride-1 shortcut)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-027.md
- **Plan**: plans/plan-027.md
- **Log**: logs/exp-log-027.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) within a fixed 300s training-compute budget on a single H20, editing only `train.py`. Baseline = **96.22** (EXP-012); pass bar = **96.32** (+0.1pp). EXP-027 tests whether an information-preserving residual downsample lifts top-1.

## Idea & Hypothesis
Chosen idea: **ResNet-D downsample** (He et al., "Bag of Tricks", CVPR 2019). The standard residual downsample uses a stride-2 1×1 conv on the shortcut, which samples every other pixel and discards 3/4 of the input feature map. ResNet-D inserts a 2×2 average-pool (stride 2) *before* a stride-1 1×1 conv, so all input pixels contribute. It was chosen as the **compute-neutral re-test of EXP-024's masked hypothesis**: BlurPool (anti-aliased downsampling) regressed to 95.66 but was COMPUTE-CONFOUNDED (it moved the heavy 3×3 main conv to stride-1, 4×'ing FLOPs → epochs 91→77). ResNet-D touches only the cheap 1×1 shortcut (same conv FLOPs at the pooled resolution + a trivial avgpool), so it should hold ~91 epochs and finally test the "don't discard pixels at downsampling" idea fairly. Hypothesis: preserving residual-path information lifts `best_test_acc` above 96.32 at unchanged ~91 epochs / 4,299,866 params.

## Approach
Single architecture change in `BasicBlock.__init__` (train.py L80-86): replaced the two-way shortcut branch with a three-way branch — `stride != 1` → `AvgPool2d(2,stride=2) → Conv2d(in,out,1,stride=1,bias=False) → BatchNorm2d(out)` (the 2 downsample blocks layer2.0/layer3.0); `in≠out` only (stride==1) → plain stride-1 1×1 + BN (layer1.0 channel-change); else `Identity`. No config changes (LR 0.2, batch 128, warmup 0.05, WD 1e-4, LS 0.1, Cutout 16, TA, cosine-to-0 all unchanged). Smoke test confirmed params unchanged at 4,299,866 (avgpool param-free, 1×1 shape identical), forward (2,3,32,32)→(2,10), and exactly layer2.0/layer3.0 on the ResNet-D path with layer1.0 on the plain 1×1. No deviations from plan.

## Execution
Single run, no retries. Launched `CUDA_VISIBLE_DEVICES=0 uv run train.py` on GPU 0, exited 0 in 404.6s wall-clock. Clean compile — no graph break on AvgPool2d, no NaN/Traceback. A Monitor watch reported the summary block on completion.

## Results
- **Primary metric**: 95.75% (baseline: 96.22, delta: **−0.47pp**, −0.49%)
- **Observations**: num_epochs 89, num_steps 34562 (baseline ~91 / ~35500), mean dt ≈ 8.7ms, peak_vram 467 MB. **final_test_loss 0.2099 — WORSE than baseline 0.195.** Epochs held in the throughput-neutral band (89 ≫ 85), so the avgpool's trivial compute overhead (2.6% fewer steps) does NOT explain a 0.47pp drop.
- **Analysis**: Hypothesis FALSIFIED, and this is a **fair, compute-neutral test** (unlike EXP-024's 77-epoch crater). The information-preserving avgpool downsample genuinely HURTS on this shallow 2-downsample CIFAR net — and uniquely it degraded BOTH top-1 and loss (the convergence-polish nulls EXP-006/019/020/026 all *improved* loss). Mechanism: the avgpool is a low-pass filter that blurs discriminative high-frequency content at the 2 downsample shortcuts, whereas the lossy stride-2 1×1 acts as a *learned* strided subsample the network adapts to. On ImageNet ResNet-50 (many downsample stages, large images) averaging helps; on a shallow 32×32 net with only 2 downsample points, the strided 1×1 is already near-optimal and forced averaging destroys signal. This also retroactively explains part of EXP-024: BlurPool's anti-aliasing component (separate from its FLOPs confound) was likely *also* unhelpful here. The downsampling/anti-aliasing axis is now closed from both the expensive (EXP-024) and cheap (EXP-027) sides.
- **Key Learning**: On this shallow CIFAR k=4 WRN, the lossy stride-2 1×1 shortcut is *better* than an information-preserving avgpool downsample — anti-aliased/lossless downsampling does not generalize from deep-ImageNet to shallow-CIFAR and hurts both top-1 and loss.

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.32) FAILED — 95.75 < 96.32; remaining conditions still recorded as informational and PASSED (clean completion, total_seconds 404.6 < 600, train.py-only, params 4,299,866 unchanged, 89 evals for 89 epochs, no new deps, seed 42).
- **Review Notes**: Results trustworthy. Throughput-neutral (89 epochs) ⇒ a fair test, not compute-confounded. No integrity concerns — pure architecture change through the allowed intervention class, single-forward eval unchanged.
- **Verdict**: no-improvement
- **Verdict Basis**: verification condition failure (primary metric below bar); valid, fair, well-controlled negative.

## Unexplored Avenues
- **Asymmetric/learned downsample** (e.g., a strided 1×1 *plus* a parallel avgpool branch summed, or a strided 2×2 conv shortcut): would test whether the issue is the averaging specifically vs. the subsampling — but adds params/FLOPs and risks the epoch wall; low expected value given both downsample variants now tested fail.
- **None promising for the downsampling axis** — both the expensive (BlurPool/EXP-024) and cheap (ResNet-D/EXP-027) information-preserving downsamples regress. The axis is exhausted.

## Next Steps
- The plateau is now ~20 axes closed (all scalar knobs bracketed, aug family closed, regularizers fail on convergence-bound recipe, compute-adding hits epoch wall, batch-scaling compute-bound, weight-averaging/polish moves loss not top-1, and now both downsample variants regress). The honest conclusion is **96.22 is at/near the k=4 / 300s ceiling**. Per NEVER-STOP, continue.
- **Per-channel input std-norm** (low confidence): the last untouched cheap scalar; assessed as expected null-or-mild-regression due to train/test std mismatch (eval is frozen at std=(1,1,1)). A quick closer.
- **More radical architecture within the param/compute envelope** (medium-low confidence): e.g., a different width/depth trade at fixed ~4.3M params and fixed 300s (e.g., k=3 deeper vs k=4 shallow), or a stem change — but width/depth was largely bracketed (k≥5 hits epoch wall). Reconsider combining two near-neutral levers if any positive signal emerges.

## Exit Action Results
<!-- No exit actions defined in the goal file. -->
- None defined.
