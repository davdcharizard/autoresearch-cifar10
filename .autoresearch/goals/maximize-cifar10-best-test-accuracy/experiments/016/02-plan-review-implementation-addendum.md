# External Claude Implementation-Addendum Review - EXP-016

The mandatory external Claude review of `train.py`, `preflight_bf16.py`, and `timing_bf16.py` completed successfully. No fallback reviewer was used.

## Concerns Found

1. The timing controller checked only equal backend state and could falsely pass if cuDNN TF32 were consistently disabled across all arms.
2. Tight numerical comparisons covered reset one-step probes, while the 200-step trajectory used only loose loss/concentration checks on 25 repeated batches.
3. Requiring BatchNorm autocast output to be BF16 contradicted the proposal's instruction to observe rather than assume undocumented policy and could falsely fail.
4. Wall projection used the minimum exposure floor, but a faster candidate produces more epochs and dense-tail evaluations; this was the wrong conservative direction.
5. Stage-savings attribution was ill-conditioned near zero total savings, and BF16 stage measurement used two autocast regions unlike production's one.
6. Eval-count modeling is approximate and child imports incur unused evaluator initialization; these are secondary given the large wall margin.

## Corrections

- Assert the reviewed environment defaults in every timing arm: cuDNN TF32 on, matmul TF32 off, cuDNN benchmark off, contiguous parameters, and otherwise identical state.
- Run the 200-step trajectory on distinct production batches and record loose checkpointed logit/gradient/update/loss alignment at steps 25/50/100/150/200. This remains a collapse/drift veto, not an accuracy proxy.
- Record BatchNorm output dtype without gating it; continue gating persistent FP32 state and accumulated running-stat alignment.
- Project evaluator count from the maximum of accepted exposure and both candidate projections.
- Keep a single BF16 autocast region in staged timing and require at least 5% positive total stage savings before evaluating the 90% forward/backward attribution fraction.

A focused external Claude re-review of these corrections is required before executing either controller.

## Focused Re-review

Claude re-read the corrected full controllers and returned **APPROVED**, confirming all six original issues were mechanically addressed and no blocking controller correctness bug remained. It recommended reconciling one remaining false-fail risk: strict `0.98` logit-cosine / `1.15x` loss thresholds compared two independently diverged models after 200 high-LR steps despite the check being a collapse veto.

That recommendation is adopted: the cross-trajectory FP32-eval thresholds now match the loose drift-veto framing (`>=0.90` cosine, `<=1.50x` loss) and use five clean weak hard-label batches rather than CutMix/RandAugment batches. Two conservative wall refinements are also adopted: project the maximum permitted one evaluation per epoch, and add measured training-loader/model setup to fresh evaluator startup. These changes can only reduce false failure or make the wall gate more conservative; they do not relax any numerical one-step, funding, exposure, stage, memory, lifecycle, or production success condition.

Claude performed a final narrow review of the exact corrected source and returned **APPROVED**: the weak held-out loaders are fully shut down, the drift-veto thresholds match their purpose, result keys match, and the max-step/one-eval-per-epoch wall arithmetic is conservative. The exact reviewed controllers are now authorized to run.
