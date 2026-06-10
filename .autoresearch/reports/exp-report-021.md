# Report EXP-021: Larger Cutout (CUTOUT_SIZE 16 → 20)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-021.md
- **Plan**: plans/plan-021.md
- **Log**: logs/exp-log-021.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, by editing only `train.py` within a fixed 300s training budget on one H20. Baseline = **96.22%** (EXP-012, commit 6c417a4); success bar = baseline + 0.1pp = **96.32%**.

## Idea & Hypothesis
Chosen idea: raise the Cutout hole size `CUTOUT_SIZE` from 16→20px (≈25%→≈39% occluded area) on the EXP-012 recipe — a single-constant, compute- and param-neutral change. Selected because augmentation is the only mechanism that has produced gains on this project (Cutout EXP-002/003, TrivialAugment EXP-012), and the aug-strength UP-direction was the one lever the project's own learnings explicitly flagged as indicated-but-untested: EXP-013 (Cutout 16→8) under-regularized (acc fell, loss rose), so the recorded insight pointed to "probe larger Cutout (≥16px)". Hypothesis: stronger occlusion reduces the residual generalization gap and lifts best_test_acc above 96.32; alternatively, if acc falls / loss rises, 20px over-occludes and the Cutout optimum is ≤16 (axis closes).

## Approach
Single-line change to `train.py` line 28: `CUTOUT_SIZE = 16` → `CUTOUT_SIZE = 20`. Everything else identical to the EXP-012 baseline (k=4 WideResNet 4.3M params, PEAK_LR 0.2 cosine-to-0, batch 128, Nesterov SGD, WD 1e-4, label smoothing 0.1, TrivialAugmentWide + vectorized GPU Cutout, torch.compile reduce-overhead, bf16, channels_last, seed 42). The `cutout_batch` mask clips the hole to the image border, so the larger size needed no other code change. Lint (ruff) + AST clean; `git diff --name-only` = train.py only. No deviations from plan.

## Execution
Single run, no retries. Launched `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background). Clean startup: params 4,299,866 (unchanged), clean compile, no traceback, no NaN. Throughput-neutral at 8ms/step ~15.6k img/s — the Cutout size change adds no compute (GPU masked_fill). Completed 92 epochs in 408.1s wall-clock (300.0s training), peak VRAM 453.8MB.

## Results
- **Primary metric**: best_test_acc = **95.96%** (baseline: 96.22, delta: **−0.26pp**, −0.27%)
- **Observations**: final_test_loss = **0.1969**, ROSE vs baseline 0.195 — the exact diagnostic the plan flagged for over-occlusion. final_test_acc 95.75%, 92 epochs (epoch-count matched baseline, so this is a fair same-budget comparison, not an under-training confound).
- **Analysis**: Hypothesis REFUTED. 20px (≈39% occluded area) removes too much signal per image, degrading both top-1 and test loss. Combined with EXP-013 (8px also worse, loss rose), this brackets the Cutout optimum: both directions away from 16 hurt → **16px is a clean interior optimum** for the occlusion-strength axis under TrivialAugment. This mirrors EXP-016/017's LR-peak result (0.2 interior optimum, both directions regress) — the EXP-012 recipe's heuristic constants are already well-tuned.
- **Key Learning**: Cutout 16→20 over-occludes (acc −0.26pp, test loss rose 0.195→0.197); bracketed with EXP-013's 8px regression, 16px is a clean interior optimum and the occlusion-strength axis is CLOSED.

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.32) FAILED (95.96); Conds 2–3 skipped per protocol (would have passed — clean 408s run, scope = train.py only, params 4,299,866, eval-count 92 == epochs).
- **Review Notes**: Results confirmed trustworthy — clean exit, params/scope intact, single-constant change, loss diagnostic internally consistent with the over-occlusion mechanism. No integrity/reward-hacking concerns.
- **Verdict**: no-improvement
- **Verdict Basis**: necessary condition failure (primary metric below bar and below baseline).

## Unexplored Avenues
- **Intermediate Cutout (18px)**: untested midpoint between the 16px optimum and the 20px regression. Very unlikely to beat 16 given both neighbors are worse (interior optimum), so low value — would only refine the curve, not break the plateau. Not worth a run.
- **Cutout count (2× smaller holes vs 1× large)**: a different occlusion *structure* rather than size, but same label-preserving-occlusion family as the now-closed strength axis; low expected value.
- The augmentation family as a whole is now well-mapped: policy (TA≈RA, EXP-014), label-mixing (Mixup/CutMix fail, EXP-011/018), occlusion-strength (16 optimal, EXP-013/021). TA + Cutout(16) is the augmentation ceiling for this model/budget.

## Next Steps
- **Per-channel input std-normalization** (std=(1,1,1)→true CIFAR std ≈(0.247,0.243,0.261)) — the last untried cheap single-knob probe; confidence LOW that it helps (first layer is Conv→BN, which almost certainly absorbs a per-channel rescale → expected null), but it cleanly closes the input-normalization axis in one run. (medium confidence it's a clean null; low confidence it gains.)
- After the std-norm probe, ~14 axes are exhausted and the honest scientific call is that the **96.22 plateau is generalization-bound at fixed k=4 capacity in 300s**. Remaining moves would be more radical architecture changes (e.g. anti-aliased downsampling / BlurPool, Zhang 2019 — recorded in Unexplored Avenues) that risk the EXP-015 compute-confound. (low confidence any single knob breaks 96.32.)

## Exit Action Results
<!-- No exit actions defined for this goal. -->
