# Proposal: EMA-Aligned Bounded Late LR Floor-Cycle

## Summary

Keep the complete EXP-011 WRN/CutMix/SAM/full-state-EMA package and change only the late learning-rate function. Preserve the parent warmup/cosine value everywhere, but from progress 0.75 form an upper envelope with a small four-cycle sinusoidal floor whose period equals the EMA half-life:

```python
LATE_LR_START = 0.75
LATE_LR_FLOOR = 0.006
LATE_LR_PEAK = 0.014
LATE_LR_CYCLES = 4

def learning_rate(progress):
    base_lr = parent_learning_rate(progress)
    if progress < LATE_LR_START:
        return base_lr
    tail = (progress - LATE_LR_START) / (1.0 - LATE_LR_START)
    cycle_lr = LATE_LR_FLOOR + (LATE_LR_PEAK - LATE_LR_FLOOR) * (
        math.sin(math.pi * LATE_LR_CYCLES * tail) ** 2
    )
    return max(base_lr, cycle_lr)
```

No optimizer, averaging, augmentation, model, batch, seed, or evaluation scalar changes. This is one fixed schedule, not a sweep over floor, peak, cycle count, phase, or EMA horizon.

## Rationale

EXP-011 improved EXP-004 from 95.40% to 95.61% by adding a sparse, charged-time, full-state EMA, but its last 16 EMA checkpoints averaged 95.493 and ended at 95.46. The formal child threshold is 95.71. The current EMA is valid and nontrivial: 160 balanced samples, 18.75-second half-life, 1.51% final parameter distance, and negligible throughput cost. The open question is whether the strongly collapsing parent cosine leaves enough late trajectory diversity for averaging to find a better center.

SWA literature links useful averaging to a late LR regime that keeps iterates moving around a basin rather than converging to nearly the same point. The proposal does not replace the successful EMA with canonical uniform SWA or a constant/cyclic training regime. It adds only a bounded late floor-cycle under the existing EMA, retains a low final LR, and never lowers the validated parent schedule.

The exact values are fixed mechanistic hypotheses. `0.006` is three times the parent's final `0.002` but only 17.7% of the LR at the 0.75 transition. `0.014` is seven times the final parent LR but only 41.2% of the transition LR. The range is large enough to move the trajectory but remains far below peak LR 0.2 and below the parent LR until very late.

Sources:

- `knowledge/papers/stochastic-weight-averaging.md`
- `knowledge/papers/how-to-scale-your-ema.md`
- `experiments/011/04-analysis.md`
- `experiments/012/01-brainstorm.md`
- `02-system-understanding.md`
- `03-experiment-learnings.md`

## Continuity and Charged-Time Math

The parent LR at progress 0.75 is:

```text
q = (0.75 - 0.05) / 0.95 = 14/19
base_lr(0.75) = 0.0339491244
cycle_lr(0.75) = 0.006
```

The maximum therefore selects the parent in an open neighborhood around 0.75. The candidate is value-continuous and has the same derivative as the parent at the CutMix/SAM/EMA transition; no LR jump is introduced there. The maximum creates a derivative kink only at the later crossover where both continuous curves are equal.

Since progress is charged time divided by 300 seconds, tail coordinate 1.0 spans exactly 75 charged seconds. `sin^2(pi * 4 * tail)` has four periods, so each nominal period is:

```text
75 / 4 = 18.75 charged seconds
```

This exactly matches `EMA_HALF_LIFE_S=18.75`. Nominal cycle peaks occur at progress `0.78125`, `0.84375`, `0.90625`, and `0.96875`; troughs occur at `0.75`, `0.8125`, `0.875`, `0.9375`, and `1.0`.

The parent remains above the cycle through the first two nominal cycles and crosses the envelope at approximately progress `0.887525` (266.26 charged seconds). The schedule is therefore unchanged for the first 88.75% of training and actively floor/cycles only for the last approximately 33.74 seconds, covering about 1.8 effective cycles. Key values are:

| Progress | Parent LR | Candidate LR |
|---:|---:|---:|
| 0.75000 | 0.033949 | 0.033949 |
| 0.84375 | 0.014924 | 0.014924 |
| 0.87500 | 0.010338 | 0.010338 |
| 0.90625 | 0.006720 | 0.014000 |
| 0.93750 | 0.004107 | 0.006000 |
| 0.96875 | 0.002528 | 0.014000 |
| 1.00000 | 0.002000 | 0.006000 |

Numerical integration over charged progress gives parent versus candidate mean tail LR `0.0128984` versus `0.0156292`, a 21.17% increase. Across the full 300-second schedule, integrated LR dose rises only 0.673% (`30.4350` to `30.6398` LR-seconds). The candidate ends at its floor rather than at a cycle peak, preserving a bounded final refinement state.

## Interaction with SAM

Preserve `SAM_RHO=0.05`, start 0.75, and period two exactly. Both ordinary and SAM-derived optimizer updates use the same time-indexed candidate LR. The SAM perturbation remains a radius-0.05 function of the first gradient; LR is applied only by the sole Nesterov optimizer step after exact perturbation restoration. Do not scale rho with LR or alter the second loss, CUDA RNG replay, BatchNorm suppression, snapshot coverage, or momentum.

The cycle increases the displacement of late SAM and ordinary updates together. Since period two is step-based while the LR cycle is charged-time-based, throughput determines the exact number of each update per cycle, but every interval contains thousands of alternating steps. Existing SAM counters and first-progress audits remain the source of truth.

The main risk is that SAM plus a reheated floor prevents the clean tail from settling, increasing loss or pushing the online trajectory between basins. The upper envelope limits that risk: it never raises LR above 0.014 after it becomes active, keeps all prior training unchanged, and ends at 0.006.

## Interaction with the Existing EMA

Preserve the EXP-011 `ChargedTimeEMA` implementation byte-for-byte except for configuration output that identifies the LR schedule. In particular:

- start at progress 0.75;
- sample every 31st post-optimizer state;
- keep the time-derived 18.75-second half-life and per-interval decay;
- retain exact 80/80 ordinary/SAM balance up to one-sample tolerance;
- average all floating parameters and persistent floating buffers and copy integer buffers;
- evaluate live before activation and one swapped EMA state afterward;
- preserve exact restore, optimizer identity, RNG, mode, coverage, finite-distance, and BN audits.

Cadence 31 should produce roughly 40 EMA samples per nominal 18.75-second cycle and, because it is odd relative to SAM period two, balance samples from both training paths. Matching cycle period to half-life gives the immediately previous cycle 50% retention at the same phase, allowing the EMA to combine different cycle phases without an extra forward.

The active floor-cycle begins about 33.74 seconds before the end, or 1.80 EMA half-lives. At budget exhaustion, pre-cycle shadow content is therefore retained at approximately:

```text
2 ** (-33.74 / 18.75) ~= 0.287
```

and roughly 71.3% of final EMA mass comes from the diversified window. This is enough for the intervention to matter while retaining a substantial anchor to the already successful parent trajectory.

The experiment is package-level. A gain can result from the changed online solution, greater EMA diversity, their interaction, altered BN state, or a different selected checkpoint. It cannot prove that diversity alone caused the change because live tail checkpoints remain intentionally unevaluated.

## Implementation Scope

Modify only `train.py`:

1. Add the four fixed late-LR constants.
2. Preserve the parent LR calculation as an explicit helper and return the envelope above from `learning_rate`.
3. Extend startup config with `late_lr_mode=ema_aligned_floor_cycle`, start/floor/peak/cycles/period.
4. Add schedule-only audit accumulators; do not alter the EMA state machine, model, optimizer, data, or evaluator.

All scalar schedule calculation occurs inside the existing charged step and adds no GPU kernel, parameter, buffer, RNG draw, or model operation. `num_params` and expected VRAM remain EXP-011's 2,748,890 and approximately 1,222 MiB.

## Schedule and Mechanism Audit

Track both `base_lr` and applied `lr` on every step and print:

- first/last tail progress and LR;
- first override step/progress/time;
- override step count and charged seconds;
- min/mean/max applied LR in the full tail and override region;
- time-weighted parent/candidate tail dose `sum(lr * dt)` and ratio;
- nearest observed step/progress/LR to each nominal peak/trough;
- final LR and number of nonfinite/bound violations.

Use charged `dt`, not an unweighted step average, for dose. Require no override below 0.75, applied LR `>= base_lr`, applied LR `<=0.0339492` throughout the tail, cycle component in `[0.006,0.014]`, first override near `[0.8870,0.8881]`, final LR near 0.006, and actual tail-dose ratio near the continuous prediction 1.2117 within a tolerance justified by discrete boundary crossing.

Reuse the existing EMA audit and compare against the EXP-011 reference:

- EMA updates 160, expected at least 145;
- ordinary/SAM sample difference at most one;
- parent consecutive parameter-distance mean 0.36667874;
- parent final EMA/live relative parameter distance 0.01506322;
- parent late EMA mean/range/final 95.4931 / 95.44-95.61 / 95.46.

A higher distance is evidence that the schedule diversified the sampled trajectory, not evidence of improvement. Predeclare a mechanism expectation of consecutive-distance mean above 0.40 and final relative distance in `[0.015,0.05]`; report failures honestly but do not abort a numerically valid run or tune the schedule from them.

## Feasibility and Smokes

1. **Closed-form landmarks:** Evaluate the function at transition, four peaks, five troughs, and endpoint; require the table above and exact parent equality before 0.75.
2. **Dense-grid bounds:** Over at least one million progress points, require continuity to numerical tolerance, one first crossover near 0.887525, no candidate value below parent, no active value above declared bounds, and exactly four nominal sinusoidal periods.
3. **Integration:** Numerically rederive tail means/dose ratio 1.2117 and full-run dose ratio 1.00673 independently from production code.
4. **Parent-prefix parity:** Run fixed-seed parent/candidate training through representative progress values below 0.8875 and require identical LR, live model/optimizer/BN state, losses, outputs, RNG, SAM restoration, and EMA shadows.
5. **Active-tail divergence:** At peak/trough landmarks, require only optimizer LR and consequent trajectory/EMA state to differ; CutMix remains off and SAM/cadence decisions remain identical for the same step/progress inputs.
6. **EMA integrity:** Re-run full-state inventory, charged decay, 31-cadence parity, exact successful/exceptional swap restoration, optimizer identity, RNG, BN, and one-evaluation checks unchanged.
7. **Numerical GPU smoke:** Run BF16/channels-last ordinary/SAM pairs through two cycle peaks and the final floor; require finite losses/gradients/parameters/buffers, perturbation norm 0.05, and no momentum or BN anomaly.

The schedule adds only CPU scalar math, so a long latency benchmark has little value. A paired GPU-0 preflight should still run five alternating production-faithful rounds and require candidate/parent weighted median latency ratio `<=1.005`, projected steps `>=25,200`, peak allocation `<1.30 GiB`, projected total runtime `<600s`, and parent trial dispersion `<=5%`. The parent passes relative gates by construction. Any overhead must remain inside the existing charged interval; do not move auditing or EMA work to rescue exposure.

## Risks

- **Validated convergence is disturbed:** EXP-011's parent reaches low final loss with a collapsing cosine. Raising the last 11.25% may sacrifice class-boundary refinement or final calibration.
- **EMA diversity is not quality:** Greater sample distance can average across worse or distinct basins, producing a poor central model despite stronger motion.
- **Only 1.8 cycles are active:** The envelope intentionally hides the first two nominal cycles under the parent. This is conservative but may provide too little intervention to move the stable plateau.
- **Schedule/EMA phase coupling:** Matching cycle period to half-life is mechanistically clean but not literature-validated as optimal. Cadence steps drift in phase with variable throughput.
- **Higher LR and SAM may interact badly:** Fixed rho with larger optimizer displacement can make late perturbation-aware updates oscillatory.
- **BatchNorm lag:** A more mobile online model changes running statistics; full-state EMA handles them consistently but still approximates statistics for averaged weights without recalibration.
- **Selection noise:** Parent best 95.61 sits above a 95.493 plateau. A single new maximum near 95.71 may still be weaker evidence than a raised tail mean/final.
- **No causal isolation:** The child changes the online schedule and the trajectory supplied to EMA together; it is not a pure averaging or pure LR test.

## Expected Effect and Falsification

The proposal plausibly clears 95.71, but with low-to-medium confidence. SWA evidence supports maintaining late trajectory diversity, the existing EMA is demonstrably useful and under-distant from live by only 1.51%, and the schedule changes just the final 33.7 seconds. Against that, the parent already has a stable 95.49 EMA plateau and no direct evidence establishes these floor/peak values under SAM.

The preregistered prediction is `best_test_acc = 95.71-95.86%`, with a supporting target of final-16 EMA mean `>=95.65%` and final accuracy `>=95.60%`. Formal success requires at least 95.71 versus parent 95.61; the supporting plateau targets distinguish a stable gain from another selected maximum but do not replace the formal metric.

After preflight, run exactly once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

Require physical GPU 0 to be the 97,871 MiB H20, exit 0, 299.5-301.0 charged seconds, total time below 600 seconds, at least 25,200 steps, unchanged 2,748,890 parameters, complete summary, one evaluation per epoch, unchanged CutMix/SAM/EMA cadence and state integrity, valid schedule/dose audit, and no nonfinite, restoration, RNG, coverage, CUDA, OOM, or timeout failure.

The experiment is falsified by `best_test_acc <95.71`, preflight/schedule-integrity failure, fewer than 25,200 steps, or any hard-constraint violation. A threshold pass without improved tail mean/final is a formal improvement but weak support for the diversity hypothesis. Do not rerun, change floor/peak/cycles, alter the EMA horizon/cadence, or choose a schedule from intermediate/final test accuracy.
