# Report EXP-007: Aligned width 6x on the compiled recipe
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-007.md
- **Plan**: plans/plan-007.md
- **Log**: logs/exp-log-007.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397 (EXP-006: torch.compile on the 4x doubly-regularized recipe). Specific question: with throughput unlocked by compile (139 epochs at 4x) and the H20 channel-alignment rule satisfied by construction, does a 6x-wide model (96/192/384) finally convert capacity into accuracy where 8x (EXP-002) and 5x (EXP-005) failed?

## Idea & Hypothesis

Chosen idea: WIDTH_MULT 4 → 6 with everything else byte-identical — the canonical aligned capacity step. It was the strongest case ever assembled for a width retry: EXP-006 had just demonstrated the exact mechanism the retry depends on (extra epochs convert into accuracy even at "converged" 4x), and both prior width failures were explained by since-removed factors (no compile in EXP-002/005; unaligned channels in EXP-005). Hypothesis: best_test_acc ≥ 96.85, predicated on a projected 75–80 epochs (139 / 2.25^0.76 using the eager sublinear scaling exponent) clearing the ~70-epoch starvation floor. The pre-declared failure signature — depressed accuracy with final≈best at the realized epoch count — would close the width direction conclusively.

## Approach

Single constant change in train.py: `WIDTH_MULT = 4` → `6` (stage widths 96/192/384, all multiples of 32 per the project-insights High alignment rule; ~9.6M params vs 4.29M). The compiled recipe (torch.compile + pre-loop warmup, TrivialAugmentWide, RandomErasing, time-keyed one-cycle peak 0.4, batch 512, selective WD, label smoothing) unchanged. The plan carried an explicit Failed-Approach Retry Justification for re-attempting the count-2 "capacity without throughput" failure. No deviations from plan; ruff clean; only train.py modified.

## Execution

One run, no retries (task b3rwhznuk, GPU 0, `uv run train.py > run.log 2>&1`). Launch was clean — params 9,636,202 matched prediction, epoch-1 eval 37.87% healthy. The decisive deviation appeared early: **dt 57ms vs the predicted 47–55ms window** — 2.59x the 4x step time for 2.25x the FLOPs, i.e. compiled width scaling is slightly superlinear, not the sublinear exponent extrapolated from eager EXP-002. That projected ~54 epochs, below the ~70 starvation floor. It stayed under the 70ms anomaly/abort bar, so per plan the run was allowed to finish for the datapoint. Run completed cleanly: 55 epochs, total 395.4s, startup 20.9s, VRAM 2411.8 MB.

## Results

- **Primary metric**: best_test_acc = 96.00% (baseline: 96.71, delta: −0.71pp, −0.73%)
- **Observations**: final_test_acc 95.97 ≈ best 96.00 — the exact pre-declared undertraining signature; the schedule completed but the model was still improving when time ran out. Throughput ~9.0k img/s at 6x vs 22.9k at 4x compiled. Startup only 20.9s (compile cost did not grow with model size). VRAM 2.4GB — capacity was never memory-bound, only time-bound.
- **Analysis**: The hypothesis hinged on the eager-derived sublinear scaling exponent (~0.76) carrying over to compiled execution; it did not. The compiled 4x baseline (22ms) is unusually efficient — inductor fusion gains are largest exactly where the experiment needed them to be smallest — so 6x paid ≈ linear-or-worse FLOPs cost and landed at 55 epochs, deep in the starvation zone. This is the third width failure (8x@40ep eager, 5x@52ep unaligned, 6x@55ep aligned+compiled), and the epoch counts line up cleanly: every attempt below ~70 epochs lost, regardless of alignment or compile. The width direction is now closed at the current throughput regime — not because capacity doesn't help (WRN prior still stands), but because no available width step keeps epochs above the floor.
- **Key Learning**: Compiled throughput scales ~linearly-or-worse in FLOPs across width (2.59x time for 2.25x FLOPs) — eager scaling exponents do not transfer; any future capacity step must first demonstrate ≥70 projected epochs from *measured compiled* dt.

## Verification

- **Conditions**: condition 2 failed (best_test_acc 96.00 < 96.81 = baseline + 0.1pp); condition 1 passed (clean exit, 395.4s ≤ 600); condition 3 skipped per first-failure stop (informally compliant: 55 eval lines = 55 epochs)
- **Review Notes**: results confirmed trustworthy — metric read directly from run.log summary, cross-checked against eval lines; training_seconds exactly 300.0 (budget integrity); params match the 6x architecture; no constraint violations (single GPU 0, one eval/epoch, only train.py touched)
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid run, metric below baseline + 0.1pp)

## Unexplored Avenues

- **mode="max-autotune" compile + width**: a few % more img/s would not bridge 55 → 70 epochs (needs ~27% more throughput); only worth bundling if a future lever finds ~1.3x.
- **Wider-but-shallower (NUM_BLOCKS 2, width 6x)**: trades depth FLOPs for width at roughly constant time; could keep epochs near the floor while raising capacity — the one untested topology direction for capacity. Risk: ResNet-14 depth may itself cost accuracy.
- **Intermediate 4x→5x-equivalent via wider final stage only**: non-uniform widening (e.g. 64/128/320) concentrates params where features are cheapest (8x8 maps); unexplored and alignment-compatible.

## Next Steps

1. **Mixup or CutMix at the 139-epoch 4x schedule** — the strongest remaining lever: regularization re-promoted by EXP-006's epoch unlock, published +0.5–1pp on CIFAR WRNs, zero throughput cost. Confidence: medium.
2. **compile mode="max-autotune"** — cheap throughput probe; standalone gain likely sub-bar but the measured dt updates the (now-closed) capacity calculus. Confidence: low standalone.
3. **Wider-but-shallower topology (NUM_BLOCKS 2 + aligned width)** — the only capacity path not foreclosed by the epoch floor, since it adds width without adding time. Confidence: low-medium.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
