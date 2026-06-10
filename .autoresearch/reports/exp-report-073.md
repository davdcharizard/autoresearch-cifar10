# Report EXP-073: Short Clean Warmup Before CutMix
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-073.md
- **Plan**: plans/plan-073.md
- **Log**: logs/exp-log-073.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed harness while modifying only `train.py`. The active baseline was 94.11% at commit `1119ff8`, and the goal requires at least +0.10 percentage points, so EXP-073 needed `best_test_acc >= 94.21%`.

## Idea & Hypothesis
The chosen idea was to delay CutMix for the first 2000 updates, then restore the validated static CutMix recipe. The hypothesis was that early mixed-label spatial noise might slow feature formation, while preserving full post-drop CutMix would keep the strongest known regional regularization behavior.

## Approach
`train.py` added `CUTMIX_WARMUP_STEPS = 2000`, printed the warmup setting at startup, and printed a one-time marker when CutMix became enabled. CutMix sampling was gated by `step >= CUTMIX_WARMUP_STEPS`; all other anchor settings were unchanged, including `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, `CUTMIX_LABEL_SMOOTHING=0.05`, `LR_MILESTONES=[21000, 64000]`, batch size 128, unit-std normalization, reflection crop padding, compile/channels-last, and validation once per epoch.

## Execution
One local foreground run was launched on GPU0 with `env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`. Preflight syntax and style checks passed. The startup log confirmed the CutMix anchor and `CutMix warmup steps: 2000`; the runtime marker printed immediately after step 2000. The first LR drop was reached at step 21000 in epoch 54, and the run completed cleanly with no CUDA, import, NaN, or infrastructure errors.

## Results
- **Primary metric**: 94.14% (baseline: 94.11%, delta: +0.03pp, +0.03%)
- **Observations**: Early clean warmup reached 78.06% by epoch 5; after CutMix enabled, epoch 6 reached 81.41%. Post-drop accuracy climbed to 94.14% by epoch 69 but never reached the 94.21% threshold.
- **Analysis**: The hypothesis was not supported strongly enough. A short clean warmup did not harm the first LR drop or throughput, but it also did not improve beyond the static CutMix anchor by a meaningful margin.
- **Key Learning**: A 2000-step clean warmup gives only a sub-threshold gain, so the current static CutMix schedule remains preferable.

## Verification
- **Conditions**: all process conditions passed; improvement condition failed.
- **Review Notes**: Results are trustworthy. The run completed, reported numeric metrics, preserved the hard `train.py` scope, reached the first LR drop, and stayed below the 10-minute cap.
- **Verdict**: no-improvement
- **Verdict Basis**: `best_test_acc=94.14%` is valid but below the required 94.21% noise-guard threshold.

## Unexplored Avenues
- A shorter warmup such as 500-1000 steps might reduce only the most unstable initial updates, but EXP-073 suggests the binary warmup effect is small.
- A probability ramp from 0.0 to 0.5 remains distinct, but it should be lower priority because the clean hard warmup did not clear the threshold.
- Combining a temporal CutMix schedule with a separate near-miss such as fan-out initialization is possible, but isolated temporal CutMix now has weak evidence.

## Next Steps
1. Medium confidence: test a distinct mechanism that can compose with the CutMix anchor without weakening post-drop regularization.
2. Medium confidence: consider combining one sub-threshold positive signal only when the mechanisms are independent, such as initialization plus a non-CutMix change.
3. Low confidence: revisit CutMix scheduling only with a shorter or smoother schedule if future evidence points back to early regularization timing.

## Exit Action Results
