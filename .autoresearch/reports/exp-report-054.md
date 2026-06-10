# Report EXP-054: Intermittent full-strength AugMix via RandomApply(p=0.5)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-054.md
- **Plan**: plans/plan-054.md
- **Log**: logs/exp-log-054.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) by editing only `train.py` within the 300s Σdt budget on a single H20. Baseline at loop start = **96.34%** (EXP-052, 292a9e2); bar = baseline + 0.1 = **96.44%**. This loop pushed the live augmentation-diversity lever (chain count, per EXP-053) by delivering full-strength AugMix to a random subset.

## Idea & Hypothesis
Chosen idea: replace uniform `AugMix(mixture_width=2, chain_depth=1)` with `RandomApply([AugMix()], p=0.5)` — the full default AugMix (w3, d-1; 3 chains) applied to ~50% of images. Reasoning: EXP-052 showed AugMix mixing lifts top-1 and EXP-053 showed the lever is chain COUNT (not magnitude), but full w3 is wall-infeasible uniformly (~792s). Stochastic application delivers genuine 3-chain diversity to a subset at an affordable average CPU cost (~13ms/batch). Hypothesis: throughput-neutral on Σdt (dt 8ms), wall < 600s; IF richer 3-chain diversity on half the images regularizes better than uniform w2,d1, best_test_acc ≥ 96.44.

## Approach
One-line change in `train_tf`: `AugMix(w2,d1)` → `RandomApply([AugMix()], p=0.5)`. Built on the EXP-052 winner. Cutout (GPU), model, optimizer, schedule, seed 42, batch 128, compile unchanged. num_params 4,299,866. A planning-phase dataloader probe (calibrated against EXP-052, where isolated 12.6ms → actual 571.9s) gave p=0.5 = 12.9ms/batch → ~585s; p=0.4 = 11.1ms → ~517s held as a fallback. Chose p=0.5 (max coverage) with an early real-load abort gate.

## Execution
One clean run on idle GPU 1, exit 0, 593.0s wall, no retries, no NaN. Early real-load check at 11.4% projected ~535s (passed the 585s gate → no fallback to p=0.4). dt steady 8ms; ep1 47.45% (normal). The final wall (593.0s) came in notably higher than the 535s early projection — RandomApply's stochastic per-batch cost (variable # augmented samples) widens wall variance — but stayed under the 600s limit (7s margin).

## Results
- **Primary metric**: best_test_acc **96.45%** (baseline 96.34, delta **+0.11pp**, +0.11%) @ ep88; final 96.41% @ ep91. Clears the 96.44 bar by 0.01pp.
- **Observations**:
  - **final_test_loss 0.1968 < EXP-052's 0.2010** AND top-1 up — unlike the severity null (EXP-053, loss↓/top1-flat), here BOTH moved favorably, the signature of a genuine (if small) diversity gain.
  - dt steady 8ms (633×8ms / 68×9ms); num_epochs 91 = EXP-052 (Σdt budget unaffected by the CPU-side change). peak_vram 453.8 MB unchanged.
  - Wall 593.0s — tight; large run-to-run variance from RandomApply's stochastic cost (projection 535 vs actual 593).
- **Analysis**: Hypothesis confirmed, marginally. Full 3-chain AugMix on ~50% of images beat uniform w2,d1 by +0.11pp — a 3rd confirmation that augmentation DIVERSITY is the live lever, and specifically that chain-COUNT diversity (not magnitude, EXP-053) is what helps, deliverable to a SUBSET within the wall when it can't be afforded uniformly. The coverage-reduction risk (half the images get no photometric aug) did NOT wash the benefit — the richer 3-chain exposure on the augmented half outweighed it. BUT honesty demands caution: +0.11pp is within the ±0.25pp noise band and clears the bar by only 0.01pp; this is now the THIRD stacked near-noise augmentation gain (EXP-012 +0.22 → EXP-052 +0.12 → EXP-054 +0.11), cumulatively 96.00→96.45 (+0.45pp), which is more convincing in aggregate than any single step. The favorable loss movement and the validated mechanism support it being real.
- **Key Learning**: Full 3-chain AugMix on ~50% of images (RandomApply p=0.5) beat uniform w2,d1 by +0.11pp (96.45) — chain-COUNT diversity is the lever and can be delivered to a SUBSET within the wall; coverage reduction did not cancel it.

## Verification
- **Conditions**: all passed. Cond1 best_test_acc 96.45 ≥ 96.44 ✓ (+0.11pp); Cond2 clean completion, total_seconds 593.0 < 600 ✓ (tight), num_params 4,299,866 ✓, no NaN ✓; Cond3 only train.py modified, eval/prepare untouched, evaluate() once/epoch, no new deps (RandomApply/AugMix torchvision-native), seed 42 unchanged ✓.
- **Review Notes**: Results trustworthy — clean uncontended deterministic run, scope-clean, dt steady 8ms. THREE caveats logged for confidence calibration: (1) clears the bar by only 0.01pp; (2) +0.11pp is within the ±0.25pp noise band (the 3rd stacked near-noise gain — aggregate +0.45pp over 3 steps is the stronger evidence); (3) the 593s wall is tight with large variance, so a replication could approach/exceed 600s. Verdict rests on clearing the pre-registered +0.1 bar + the validated mechanism + favorable loss movement. No integrity/scope concerns.
- **Verdict**: improvement
- **Verdict Basis**: all necessary conditions passed; primary metric exceeded baseline by the pre-registered +0.1pp margin (+0.11pp).

## Unexplored Avenues
- **Tune p (e.g., 0.4, 0.6) or the augmented-branch richness**: p=0.5 was the first feasible point tried; the coverage-vs-richness tradeoff is unmapped. p=0.6 (probed borderline ~700s — likely infeasible); p=0.4 (safe ~517s) trades coverage for wall margin. A cleaner-wall p=0.4 might match/beat p=0.5 with more headroom.
- **Replicate EXP-054** to confirm 96.45 above noise (and check the tight wall reproduces under 600s) before building further — given three stacked near-noise gains, a confirmation run is increasingly valuable.
- **RandomApply with a LIGHT else-branch instead of identity** (e.g. w3 on 50%, w1,d1 on the rest) — preserves all-image coverage while still concentrating richness — was infeasible at full cost but a cheaper light branch might fit.

## Next Steps
- **Replicate the 96.45 (and the 593s wall)**: three stacked near-noise gains (96.00→96.45) warrant a confirmation run before further investment; also checks wall reproducibility under the tight 600s margin. (medium)
- **Map the p tradeoff (p=0.4 safe-wall vs p=0.5)**: p=0.4 gives ~80s more wall headroom; if it matches p=0.5's top-1 it is the more robust config. (medium)
- **The diversity lever may still have headroom via throughput recovery** (GPU-side per-sample AugMix) to afford richer uniform chains — highest upside, highest implementation risk. (low)

## Exit Action Results
<!-- No exit actions defined for this goal. -->
