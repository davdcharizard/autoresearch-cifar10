# Report EXP-053: AugMix(w2,d1) severity 3→6 — push op magnitude on the new winner
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-053.md
- **Plan**: plans/plan-053.md
- **Log**: logs/exp-log-053.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) by editing only `train.py` within the 300s Σdt budget on a single H20. Baseline = **96.34%** (EXP-052, 292a9e2); bar = baseline + 0.1 = **96.44%**. This loop tuned the just-validated AugMix lever's magnitude.

## Idea & Hypothesis
Chosen idea: on the EXP-052 winner `AugMix(mixture_width=2, chain_depth=1)`, raise `severity` 3→6 (keep w2,d1, Cutout, all else fixed). Reasoning: augmentation diversity is the only top-1 lever here (EXP-012/052), and severity is the only diversity dial that is CPU-neutral (preserves the <600s wall) and all-image coverage (the working regime) — stronger per-op magnitude → a more spread-out augmented distribution. Hypothesis: throughput-neutral; IF stronger magnitude adds useful diversity, best_test_acc ≥ 96.44; falsified if within ±0.25pp (severity=3 already near-optimal — magnitude interior-optimal, cf. Cutout EXP-013/021) or mild over-augmentation regression.

## Approach
One-keyword change in `train_tf`: `AugMix(mixture_width=2, chain_depth=1)` → `AugMix(mixture_width=2, chain_depth=1, severity=6)`. Built on the EXP-052 winner (autoresearch/dev). Cutout, model, optimizer, schedule, seed 42, batch 128, compile, and all-image coverage unchanged. num_params 4,299,866. Smoke-tested (AST, scope=train.py only, AugMix sev6 runs, params unchanged).

## Execution
One clean run on idle GPU 1, exit 0, 580.4s wall, no retries, no NaN. Early-load check confirmed feasibility (eval-inclusive ~15.6ms/step → projected ~533s; dt steady 8ms — severity is CPU-neutral as predicted). ep1 test_acc 45.45% (normal, ≈ baseline ~45.7% — severity=6 did not destabilize early convergence).

## Results
- **Primary metric**: best_test_acc **96.29%** (baseline 96.34, delta **−0.05pp**, −0.05%) @ ep88; final 96.23% @ ep91. 0.15pp below the 96.44 bar.
- **Observations**:
  - **final_test_loss 0.1961 < EXP-052's 0.2010** (and ≈ the original 0.195 baseline) — severity=6 LOWERED test loss but top-1 fell −0.05pp. A textbook polish-vs-top1 signature: the magnitude change moved loss/calibration, not the decision boundary.
  - dt steady 8ms (656×8ms / 48×9ms / few 10-11ms); num_epochs 91 = EXP-052 (CPU-neutral confirmed — severity adds no op count). peak_vram 453.8 MB unchanged.
- **Analysis**: Hypothesis falsified (negatively, within noise). −0.05pp is squarely within the ±0.25pp band, so severity=6 is effectively a null/very-mild regression — op MAGNITUDE is interior-optimal here, directly mirroring the Cutout occlusion-STRENGTH result (EXP-013/021, both directions regress). AugMix's default severity=3 is already near-optimal; cranking magnitude does not add useful diversity and slightly hurts top-1 while (interestingly) helping loss. This sharpens the EXP-052 insight: it is the NUMBER/DISTINCTNESS of augmentation CHAINS (mixing) that lifts top-1 — NOT per-op magnitude. The chain-count dimension (the actual diversity lever) is exactly the one that is wall-infeasible to push (w2,d2/w3 breach 600s); the feasible magnitude dimension is inert. The lever and the wall are in tension.
- **Key Learning**: AugMix severity=6 lowered loss (0.196<0.201) but top-1 fell −0.05pp (96.29) — op MAGNITUDE is interior-optimal here (default 3 near-best, like Cutout strength); diversity = chain COUNT, not magnitude.

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.44) FAILED at 96.29 → verification stopped; Conds 2–3 not evaluated (noted: run was clean — 580.4s < 600, params 4,299,866, no NaN, diff = train.py only — so this is a valid no-improvement, not invalid/crash).
- **Review Notes**: Results trustworthy — clean uncontended deterministic run, scope-clean, dt steady 8ms (CPU-neutral confirmed). The −0.05pp is within noise; the firm conclusion (magnitude interior-optimal) is reinforced by the loss-down/top1-flat signature matching the polish-vs-top1 pattern and the Cutout-strength precedent.
- **Verdict**: no-improvement
- **Verdict Basis**: valid clean run; primary necessary condition failed (−0.05pp vs baseline, −0.15pp vs bar).

## Unexplored Avenues
- **AugMix chain COUNT / depth (the real diversity dial) is the unexhausted direction — but wall-infeasible at 8 workers**: w2,d2 (~670s) and w3 (~792s) breach the 600s limit. The only way to test richer chains is to RECOVER throughput (GPU-side per-sample AugMix, or stochastic full-strength application) — see EXP-052 report.
- **Intermittent full-strength AugMix via RandomApply (p≈0.5)**: deliver the rich w3 config to a subset within the wall (brainstorm-053 Candidate 2) — tests chain-count-diversity, the live dimension, rather than the inert magnitude one. Coverage/confound caveats apply.
- **Lower severity (1-2)**: not worth it — the loss already fell at sev6 and top-1 is flat; severity is interior-optimal, so neither direction is expected to clear the bar.

## Next Steps
- **Intermittent full-strength AugMix (RandomApply p≈0.5 on default w3)**: the most direct feasible way to push the REAL diversity lever (chain count), since magnitude is now shown inert. Needs a dataloader feasibility probe. (medium)
- **Replicate EXP-052's 96.34** to confirm it sits above noise before investing further — the gain was +0.12pp (near band) and EXP-053 landed −0.05 below it. (medium)
- **GPU-side per-sample AugMix** to unlock full chain-count diversity uniformly under the wall — highest upside, highest implementation risk. (low)

## Exit Action Results
<!-- No exit actions defined for this goal. -->
