# Report EXP-055: Richer per-image AugMix chains on a subset — RandomApply([AugMix(mixture_width=4)], p=0.35)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-055.md
- **Plan**: plans/plan-055.md
- **Log**: logs/exp-log-055.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) by editing only `train.py` within the 300s Σdt budget on a single H20. Baseline at loop start = **96.45%** (EXP-054, 86161d9); bar = baseline + 0.1 = **96.55%**. This loop probed the one untested direction of the validated chain-count diversity lever: raising AugMix mixture_width 3→4.

## Idea & Hypothesis
Chosen idea: `RandomApply([AugMix()], p=0.5)` (EXP-054 winner, w3/50%) → `RandomApply([AugMix(mixture_width=4, chain_depth=-1)], p=0.35)` — a RICHER 4-chain AugMix mixture on ~35% of images. Reasoning: chain-COUNT diversity is the only lever that has ever lifted top-1 here (3× confirmed, EXP-012/052/054); width 3→4 is its direct untested extension. Lower p (0.35 vs 0.5) paid for the heavier per-image op to keep the dataloader cost under the wall. Hypothesis: dt steady ~8ms, wall < 600s; IF richer 4-chain diversity regularizes better than 3-chain (at the cost of lower coverage), best_test_acc ≥ 96.55. A null/regression would bound the lever at width 3 in this regime (mirroring the EXP-053 magnitude null).

## Approach
One-line change in `train_tf`. Built on the EXP-054 winner. GPU Cutout, model, optimizer, schedule, seed 42, batch 128, compile unchanged; num_params 4,299,866. The p=0.35 (not the brainstorm's nominal 0.4) was chosen because this session's isolated dataloader probe put w4/p=0.4 at 12.2ms/batch — essentially identical to EXP-054's w3/p=0.5 (12.1ms) which shipped at the tight 593.0s wall; the EXP-054 protocol finding mandates a conservative wall target, so p=0.35 (11.4ms) was led with, p=0.3 held as a gated contingency. Smoke tests passed (AST, scope=train.py only, 10 samples shape (3,32,32), params unchanged).

## Execution
One clean run on idle GPU 0, exit 0, **543.0s wall**, no retries, no NaN. Early two-point real-load wall gate (window +34s/+2300 steps) measured 14.8ms/step → projected ~534s, +60s buffer = 594s ≤ 595 gate AND ≤540s protocol target → passed at p=0.35 (no fallback). dt steady 8ms; ep4 test_acc 73.57% (normal early). The final wall (543.0s) matched the 534s projection closely — UNLIKE EXP-054's +58s drift (535→593), p=0.35 reproduced its projection with a comfortable 57s margin, confirming p=0.35 is the safer wall config.

## Results
- **Primary metric**: best_test_acc **96.01%** (baseline 96.45, delta **−0.44pp**) — a clear REGRESSION, far below the 96.55 bar. final 96.00%; final_test_loss **0.2000** > EXP-054's 0.1968 (both top-1 and loss regressed).
- **Observations**:
  - dt steady 8ms (669×8ms / 45×9ms / 1×10ms); num_epochs 92, num_steps 35,779 (Σdt budget intact — the CPU-side aug change did not cut compute). peak_vram 453.8 MB unchanged.
  - Wall 543.0s, 57s margin — p=0.35 is genuinely safer than EXP-054's p=0.5 (593s), as planned.
  - The 96.01 result lands almost exactly at the pre-AugMix-tuning level (EXP-012-era ~96.00), i.e. the three stacked diversity gains (96.00→96.45) were effectively erased.
- **Analysis**: Hypothesis REJECTED. Two changes co-moved (confound, pre-registered): width 3→4 (richer) AND coverage 50%→35% (less). The −0.44pp regression is well beyond the ±0.25pp noise band, so it is a real effect, not noise. Attribution: the most likely dominant cause is the **coverage drop** — EXP-054 established that 50% photometric-aug coverage was the working regime, and dropping to 35% leaves 65% of images with only crop+flip(+Cutout), under-regularizing at this capacity/budget. The richer width-4 mixture did not compensate, consistent with the AugMix paper's width-saturation prior (clean-accuracy gains saturate past w3) — so width offered no upside while coverage loss imposed a real downside. Either way the verdict on the *direction* is firm: pushing width past 3 while trading down coverage is a net loss. This complements the EXP-053 magnitude null: the chain-count lever is now bounded at **width 3, coverage ~50%** as the feasible optimum in this regime.
- **Key Learning**: Width 3→4 AugMix on a reduced 35% subset regressed −0.44pp to 96.01 (≈pre-tuning level); the chain-count lever does not extend past w3 and coverage <50% under-regularizes — the EXP-054 config (w3/p=0.5) is at/near the feasible optimum.

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.55) **FAILED** — 96.01 (−0.44pp vs baseline). Per protocol, remaining conditions skipped (informationally both would pass: clean completion 543.0s < 600, num_params 4,299,866, no NaN; scope train.py-only, eval/prepare untouched, no new deps, seed 42 unchanged).
- **Review Notes**: Result trustworthy — clean uncontended deterministic run, scope-clean, dt steady 8ms, wall well under limit, metric regression far exceeds the noise band (not a false failure). No integrity/scope concerns. The confound (width↑ + coverage↓) limits clean single-variable attribution but does not affect the directional verdict.
- **Verdict**: no-improvement
- **Verdict Basis**: necessary Cond 1 failed; primary metric 96.01 regressed −0.44pp below the 96.45 baseline (−0.54pp below the bar). Valid run, no constraint violation → no-improvement (not invalid/crash).

## Unexplored Avenues
- **Width 4 at FULL EXP-054 coverage (p=0.5)** — would isolate the width effect from the coverage confound, but is wall-infeasible (w4/p=0.5 ≈ 16ms/batch → ~700s ≫ 600s). The only way to test richer width at high coverage is GPU-side AugMix (throughput recovery).
- **Coverage sweep at the validated w3** (p=0.45, p=0.55-if-feasible) to pin the coverage optimum near EXP-054's p=0.5 — cleaner single-variable than this loop; p>0.5 is wall-limited though.
- **A light else-branch instead of identity** (w3 on 50%, w1,d1 on the rest) to push coverage toward 100% without the full-cost wall hit — EXP-054 unexplored avenue, still untried.
- **GPU-side per-sample AugMix** — the only path to afford rich chains at full coverage; high implementation risk.

## Next Steps
- **Stop pushing AugMix knobs — the lever is mapped**: width (EXP-055 ✗ past 3), magnitude (EXP-053 ✗), coverage (EXP-054 p=0.5 optimal, lower hurts). The CPU-side augmentation-diversity lever is at its feasible frontier (w3/p=0.5, 96.45). Further marginal knobs are unlikely to clear the bar. (high)
- **Pivot to a genuinely new lever or the GPU-side throughput unlock**: to afford richer/fuller augmentation, move AugMix-style ops to a GPU-batched per-sample implementation (frees the dataloader wall), OR explore an untried axis (e.g. a different regularizer that composes with the saturated aug, longer-schedule via throughput, or a capacity/architecture change). Highest upside, highest risk. (medium)
- **Replicate/consolidate 96.45** if the next new-lever probe also stalls — three stacked near-noise gains still warrant a confirmation before deeper investment. (low)

## Exit Action Results
<!-- No exit actions defined for this goal. -->
