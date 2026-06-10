# Report EXP-079: Short CutMix Probability Ramp
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-079.md
- **Plan**: plans/plan-079.md
- **Log**: logs/exp-log-079.md

## Goal
Maximize CIFAR-10 `best_test_acc` under the fixed single-GPU, fixed-budget benchmark while modifying only `train.py`. The active baseline from the experiment index was 94.11% at commit `1119ff8`; with the +0.10 percentage-point noise guard, EXP-079 needed `best_test_acc >= 94.21%` to count as an improvement.

## Idea & Hypothesis
EXP-079 tested whether the current CutMix anchor is slightly too noisy during the earliest high-variance updates. The selected idea was to linearly ramp `CUTMIX_PROB` from 0.25 to 0.5 over the first 1000 optimizer steps, then restore the validated static `p=0.5` behavior for the rest of training. The hypothesis was that a short ramp would reduce early regional mixed-label noise without weakening the important post-drop CutMix regularizer, improving post-drop convergence enough to clear 94.21%.

## Approach
The implementation changed only `train.py`. It added `CUTMIX_PROB_START = 0.25`, `CUTMIX_PROB_RAMP_STEPS = 1000`, and `current_cutmix_prob(step)`, then replaced the static CutMix Bernoulli probability with the scheduled value. Step 0 used `p=0.25`; step 1000 and later used the unchanged anchor `CUTMIX_PROB=0.5`. CutMix alpha, endpoint label smoothing, clean label smoothing, stage widths, optimizer, LR milestones, reflection crop padding, unit-std normalization, compile/channels-last, batch size, seed, validation cadence, and evaluation harness were preserved. Startup logging printed the probability-ramp marker.

## Execution
One local foreground run completed on GPU0 with `CUDA_VISIBLE_DEVICES=0`. Preflight checks passed: `git diff --name-only` listed only `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` passed. Startup confirmed CUDA, `ResNet-20 | params: 822,790`, unchanged CutMix settings, and `CutMix prob ramp: 0.25 -> 0.5 over 1000 steps`. The first LR drop was reached at step 21000 in epoch 54 with about 139 seconds remaining, and the run completed normally with no traceback, CUDA, error, NaN, or non-finite signatures.

## Results
- **Primary metric**: 94.09% (baseline: 94.11%, delta: -0.02pp, -0.02%)
- **Observations**: The run followed the expected post-drop climb. Accuracy rose from a pre-drop best of 88.49% to 91.75% at epoch 54, 93.87% at epoch 63, and peaked at 94.09% at epoch 82. It finished at 93.26% after 104 epochs and 40,252 optimizer steps.
- **Analysis**: The hypothesis was not supported. The ramp produced a clean near-anchor result, but it did not beat the static CutMix baseline and stayed below the 94.21% noise-guard threshold. Together with EXP-073's 94.14% clean warmup and EXP-065/066 static probability brackets, this suggests early CutMix timing can move the metric by a few hundredths but has not shown enough leverage to count as a real improvement.
- **Key Learning**: A 1000-step CutMix probability ramp peaked at 94.09%, below baseline and threshold, so early probability ramping is not enough.

## Verification
- **Conditions**: All process and hard-constraint checks passed; the metric-improvement condition failed.
- **Review Notes**: Results are trustworthy. The run used one GPU, modified only `train.py`, preserved the fixed evaluation harness, reached the planned LR drop, produced numeric final summary metrics, and stayed under the 10-minute limit.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid run, but `best_test_acc=94.09%` is below the 94.11% baseline and below the 94.21% improvement threshold.

## Unexplored Avenues
- A shorter or smoother early CutMix schedule could still be tested, but EXP-073 and EXP-079 both suggest the effect size is small and likely noisy.
- A coupled early-stability change such as a very short LR warmup could target the same early-update instability through optimizer dynamics rather than CutMix frequency.
- More CutMix temporal weakening after the first LR drop is not supported: EXP-069 already showed post-drop probability tapering regressed to 93.73%.

## Next Steps
Medium confidence: test the very short LR warmup from EXP-079's fallback candidates, because it targets early instability without further weakening the validated CutMix anchor.

Medium confidence: revisit one of the strongest near-miss non-CutMix signals only if it is mechanistically distinct and not an additive retry already shown to fail.

Low confidence: try more CutMix probability scheduling; recent temporal and static brackets are now converging on the same near-miss band.

## Exit Action Results
