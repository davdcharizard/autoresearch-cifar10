# Report EXP-060: AutoAugment(CIFAR10 learned policy) replacing AugMix

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-060.md
- **Plan**: plans/plan-060.md
- **Log**: logs/exp-log-060.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within a fixed 300s GPU-time (Σdt) budget on a single H20, editing only train.py. Baseline = **96.45** (EXP-054); bar = **96.55** (+0.1pp).

## Idea & Hypothesis
Augmentation diversity is the only lever that has ever lifted top-1 here (Cutout +0.58, TrivialAugment +0.22, AugMix +0.12/+0.11). AutoAugment(CIFAR10) is the one untried major auto-aug POLICY family — and the only CIFAR-SPECIFIC one (25 sub-policies RL-searched to maximize CIFAR-10 accuracy). Hypothesis: the CIFAR-learned policy supplies a stronger, dataset-matched diversity distribution than the dataset-agnostic AugMix, through the same epoch-free CPU delivery, lifting best_test_acc ≥ 96.55.

## Approach
One-line augmentation swap (train.py L171): `RandomApply([AugMix()], p=0.5)` → `transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)` at full native coverage (every image gets one sub-policy), in the same Compose slot (before ToTensor, receives PIL). GPU Cutout(16) and everything else byte-identical to EXP-054 (k=4 WideResNet-20, cosine peak0.2/warmup0.05, Nesterov m0.9, WD1e-4, LS0.1, batch128, seed42, compile reduce-overhead). num_params unchanged (4,299,866). AutoAugment is core torchvision 0.24.1 — no new dep.

## Execution
Single clean run on idle GPU 1 (b8oy5he5n, exit 0). Early wall-feasibility gate (the binding risk for CPU aug) PASSED comfortably at ep11: img/s steady ~15,300 → real per-step ~8.37ms ≈ dt 8ms, i.e. the 8 dataloader workers keep up with NO starvation (AutoAugment's ~2 ops/image is far lighter than AugMix-w3's 5.8× starvation in EXP-052). Projected/actual total wall 402.5s ≪ 600s. The plan's p=0.5 coverage fallback was therefore not triggered (neither wall-infeasibility nor a clear over-reg/underfit signal — the early acc climbed healthily ep1 46.8→ep5 72.4→ep10 81.2). Full budget used: 91 epochs / 35,444 steps, dt steady 8ms (CPU aug epoch-free as designed).

## Results
- **Primary metric**: **96.22%** (baseline 96.45, delta **−0.23pp**, −0.24%)
- **Observations**: final_test_loss 0.1942 (slightly BELOW EXP-054's 0.1968), best hit ep89, converged flat tail (ep89 96.22 / ep90 96.19 / ep91 96.17) — NOT underfit, NOT a wall/epoch artifact. peak_vram 453.8 MB.
- **Decisive cross-comparison**: AutoAugment(CIFAR10) full-coverage **96.22 = TrivialAugment 96.22 (EXP-012)** to the hundredth, and both sit −0.23pp below the tuned AugMix-p0.5 (96.45, EXP-054). RandAugment was also ≈TA (96.19, EXP-014). So all three single-policy auto-augs (AA / TA / RA) cluster at ~96.2, while AugMix's multi-chain convex-mix-on-50%-subset reaches 96.45.
- **Analysis**: Hypothesis REJECTED with a clean mechanistic explanation. The premise — that a CIFAR-LEARNED policy would beat the dataset-agnostic AugMix — was the open question; the TrivialAugment-paper thesis (AA ≈ TA on CIFAR-10) is what actually obtained. The run was a fair, converged, full-budget test (91 ep, the wall was never the constraint), so 96.22 is a genuine policy-strength result, not an artifact. What distinguishes the 96.45 AugMix winner is NOT "CIFAR-specificity" or per-op magnitude but its **structure**: multi-chain Dirichlet mixing + a clean-image convex mix applied to only ~50% of images. AA/TA/RA all apply a single (1–2 op) policy at full coverage with no clean-mix — a categorically less diverse, less shift-bounded distribution. The slightly-lower loss (0.194 < 0.197) with lower top-1 is the familiar loss-not-top1 signature: AA produces marginally more confident predictions that are not more accurate.
- **Key Learning**: AutoAugment(CIFAR10) full-coverage = 96.22 = TrivialAugment (EXP-012); both < AugMix-p0.5 96.45. The auto-aug POLICY-FAMILY axis is now mapped (AA≈TA≈RA≈96.2 < AugMix 96.45) — what wins is AugMix's multi-chain-mix + 50%-subset STRUCTURE, not policy identity or CIFAR-specificity.

## Verification
- **Conditions**: Necessary condition 1 (`best_test_acc >= 96.55`) FAILED (96.22). Conditions 2 (402.5s<600, params 4,299,866, summary printed, 0 NaN/error) and 3 (scope train.py only, prepare.py/eval untouched, ≤1 eval/epoch, no new deps, seed 42, uncontended GPU 1 steady 8ms wall/Σdt~1.05×) hold.
- **Review Notes**: Trustworthy. Converged flat tail, fair full-budget run, no contention, identical epoch count to EXP-054 (confirms CPU aug epoch-free). The 96.22 = TA-96.22 match is a strong internal consistency check. No false-pass/false-fail risk.
- **Verdict**: **no-improvement**
- **Verdict Basis**: valid run, necessary condition 1 failed (−0.23pp); no hard-constraint violation.

## Unexplored Avenues
- **AutoAugment AT 50% coverage / + clean-mix**: the plan's p=0.5 fallback was not run (full coverage was wall-feasible and not clearly over-regularized). But the AugMix-vs-AA gap analysis suggests the winning ingredient is the 50%-subset + clean-mix STRUCTURE, not the policy — so AA@p=0.5 would most likely land between AA-full (96.22) and AugMix (96.45), i.e. still < bar. Low value; the structure, not the policy, is what matters and AugMix already embodies it optimally.
- **Mixing AutoAugment INTO AugMix's chains** (use AA sub-policies as the AugMix chain ops): a way to combine AA's learned policy with AugMix's mixing structure. But this is a non-trivial custom transform, likely over-regularizes (EXP-056 lesson), and the augmentation family is otherwise exhausted — very low confidence.
- The augmentation-policy avenue is now closed: AA was the last untried major policy and it confirms the family ceiling is AugMix-p0.5 = 96.45.

## Next Steps
- **The augmentation axis is now DEFINITIVELY exhausted** (high confidence): every sub-lever (strength, width, coverage, occlusion, mixing, cooldown, border) AND now every major policy family (TA/RA/AugMix/AutoAugment) is mapped; AugMix-p0.5 = 96.45 is the ceiling. Do NOT propose any further augmentation variant (CPU or GPU, any policy).
- **The plateau is mapped across EVERY standard lever**: augmentation, capacity (×4 directions), optimizer (family/objective/gradient-dynamics), schedule/LR, normalization, residual-scaling, head, batch, activation, throughput→epochs. 96.45 is at/near the achievable ceiling for this k=4 ResNet-20 at 300s.
- **Remaining genuine long-shots (low confidence)**: (a) WARMUP_FRAC scalar never isolated — a cheap clean probe, low ceiling (~noise); (b) combine the cooldown near-miss (EXP-034, never tested on the AugMix recipe) with the current best — mechanism could compound with stronger aug, but augmentation-family-closed signal makes it marginal; (c) a more RADICAL move off all closed axes (e.g. a fundamentally different micro-architecture that genuinely lowers dt to buy epochs) — but the dt-vs-capacity curve (EXP-044/058) forbids most. Per NEVER-STOP, continue principled long-shots accepting most will be no-improvement on this deeply-mapped plateau.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
- (none — no exit actions defined for this goal)
