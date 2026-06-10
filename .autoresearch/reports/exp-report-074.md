# Report EXP-074: CutMix Endpoint Hard Labels
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-074.md
- **Plan**: plans/plan-074.md
- **Log**: logs/exp-log-074.md

## Goal
Maximize CIFAR-10 `best_test_acc (%)`, where higher is better, while modifying only `train.py` and preserving the fixed evaluation harness and 300s training budget. The active baseline before EXP-074 was 94.11% at commit `1119ff8`; the goal's noise guard requires at least +0.10 percentage points, so EXP-074 needed `best_test_acc >= 94.21%` to count as an improvement.

## Idea & Hypothesis
The chosen idea was to keep the successful probabilistic CutMix anchor but remove label smoothing only from the two CutMix endpoint cross-entropy losses. The hypothesis was that CutMix already softens supervision through area-weighted label mixing, so endpoint label smoothing might over-soften mixed-batch targets while clean batches should retain the validated `label_smoothing=0.05`.

## Approach
`train.py` was changed to introduce `CLEAN_LABEL_SMOOTHING = 0.05` and set `CUTMIX_LABEL_SMOOTHING = 0.0`. The CutMix branch used the CutMix endpoint smoothing constant for both endpoint losses, while the non-CutMix branch used the clean smoothing constant. The startup marker was expanded to print both smoothing values. All anchor settings stayed fixed: `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, `STAGE_WIDTHS=(28, 56, 112)`, `LR=0.1`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection padding, unit-std normalization, compile/channels-last, and batch size 128.

## Execution
One local foreground run was launched on GPU0 with `env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Startup confirmed the expected model, CutMix anchor, and smoothing split. The first LR drop was reached at step 21000 with `lr: 0.0100`; no crash, CUDA, NaN, or non-finite-loss errors appeared. The run completed the fixed 300s training budget and reported final metrics normally.

## Results
- **Primary metric**: 94.17% (baseline: 94.11%, delta: +0.06pp, +0.06%)
- **Observations**: The run climbed to 94.02% by epoch 74, set its best of 94.17% at epoch 91, and ended at 93.19% final test accuracy. Throughput and memory matched the CutMix anchor closely: 40,577 steps, 105 epochs, and 660.4 MB peak VRAM.
- **Analysis**: The result weakly supports the idea that harder CutMix endpoints are not catastrophic, but it does not validate the hypothesis as a useful improvement. Removing endpoint smoothing produced only a sub-threshold +0.06pp gain, which the goal explicitly treats as noise.
- **Key Learning**: Hard-label CutMix endpoints can slightly exceed the baseline, but they do not clear the 94.21% guard, so endpoint smoothing is not the missing CutMix lever.

## Verification
- **Conditions**: All process and integrity conditions passed; the improvement-threshold condition failed.
- **Review Notes**: Results are trustworthy. Scope was limited to `train.py`, syntax and ruff checks passed, startup markers confirmed the intended smoothing split, and `run.log` reported numeric final metrics.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid completed run with `best_test_acc=94.17%`, below the required 94.21% threshold.

## Unexplored Avenues
- A partial endpoint smoothing value such as 0.02 or 0.025 could interpolate between hard endpoints and the anchor, but the repeated sub-threshold label-smoothing family makes this low priority.
- Hard endpoints coupled with a genuinely distinct mechanism, such as initialization or a small architecture change, might behave differently, but isolated target-regularization tweaks are no longer compelling.

## Next Steps
- Medium confidence: test a distinct mechanism outside the target-softness family, such as a lightweight architecture or initialization-plus-anchor coupling, because several target-regularization changes have now stayed inside the noise band.
- Medium confidence: revisit fan-out initialization only if coupled to an independent, low-overhead change; isolated init already reached 94.16% but also failed the guard.
- Low confidence: try a shorter CutMix warmup, but EXP-073 and EXP-074 suggest CutMix-internal refinements have small effect sizes.

## Exit Action Results
