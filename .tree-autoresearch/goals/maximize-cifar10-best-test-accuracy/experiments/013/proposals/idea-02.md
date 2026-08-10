# Proposal: Six-Half-Life Responsive Full-State EMA

## Summary

Refine EXP-011's successful charged-time EMA by shortening only its half-life from 18.75 to 12.5 charged seconds. Preserve the complete online WRN/CutMix/SAM trajectory, sparse cadence 31, full-state inventory, evaluation routing, and exact swap/restore behavior. The sole production configuration change is:

```python
EMA_START = 0.75
EMA_UPDATE_EVERY = 31
EMA_TAIL_HALF_LIVES = 6.0
EMA_HALF_LIFE_S = (
    (1.0 - EMA_START) * TIME_BUDGET_S / EMA_TAIL_HALF_LIVES
)  # 75 / 6 = 12.5 seconds
```

At each due sample after the first exact-copy initialization, retain the existing charged-time update rule:

```text
interval = current_step_entry_training_time - previous_sample_time
decay = 2 ** (-interval / 12.5)
shadow = decay * shadow + (1 - decay) * online
```

This is one preregistered horizon, not a sweep. Do not change start, cadence, online schedule, evaluation source, or any model/training scalar.

## Motivation

EXP-011 added a four-half-life, 18.75-second EMA to EXP-004 and improved `best_test_acc` from 95.40% to 95.61% with negligible charged overhead. The result is trustworthy, but its final 16 EMA checkpoints averaged 95.493, ended at 95.46, and spanned 0.17 points. Final EMA/live parameter distance was 1.506%, and averaged BN variance ratios ranged from 0.964 to 1.219. Those measurements leave two competing interpretations:

- useful smoothing remains and a longer horizon protects against checkpoint noise;
- the shadow lags the improving low-LR online solution and its BN state enough to depress the stable plateau.

The proposed 12.5-second half-life reduces lag by one third while retaining substantial smoothing. Six half-lives fit exactly in the 75-second clean tail, giving the configuration a charged-time interpretation independent of realized step throughput. The online training path is unchanged, so this experiment isolates a different evaluation-state summary more cleanly than changing LR, augmentation, or architecture.

The choice is deliberately bounded. Halving to 9.375 seconds would make the shadow much closer to live and could discard the mechanism that made EXP-011 successful; retaining 18.75 simply repeats the parent. Six tail half-lives is the smallest integer refinement between those positions with at least about 26 cadence samples per half-life under the measured parent throughput.

Sources:

- `knowledge/papers/how-to-scale-your-ema.md`
- `knowledge/papers/stochastic-weight-averaging.md`
- `experiments/011/04-analysis.md`
- `experiments/012/04-analysis.md`
- `02-system-understanding.md`
- `03-experiment-learnings.md`

## Exact Horizon and Coefficient Consequences

EXP-011 sampled 160 states over 74.7736 charged seconds, with interval min/mean/max `0.457694 / 0.470274 / 0.534709` seconds. Under the parent 18.75-second half-life, mean per-sample decay was about 0.982765. Under 12.5 seconds, expected decays at those same intervals are:

| Interval | New decay `2^(-dt/12.5)` |
|---:|---:|
| 0.457694 s | 0.974939419 |
| 0.470274 s | 0.974259555 |
| 0.534709 s | 0.970784703 |

For a fixed mean interval, the approximate geometric effective sample size `(1+decay)/(1-decay)` decreases from about 115 parent samples to about 77 candidate samples. That remains much larger than a single checkpoint.

Continuous-time mean age falls from:

```text
18.75 / ln(2) = 27.0505 seconds
12.50 / ln(2) = 18.0337 seconds
```

At budget end, the first sampled state's coefficient falls from `2^(-74.7736/18.75) = 0.063025` to approximately `0.015822`. Candidate mass from the most recent 18.75 seconds is `1 - 2^(-18.75/12.5) = 0.646447`, versus 0.5 for the parent. Thus the new shadow is more responsive but still averages several epochs and both ordinary/SAM paths.

Because the first sample initializes by exact copy, there is no zero-initialization bias and no separate bias-correction denominator. The refinement corrects temporal lag through the horizon itself rather than extrapolating beyond EMA or blending models at evaluation.

## Full-State Behavior

Retain EXP-011's `ChargedTimeEMA` semantics exactly:

- shadow all 44 trainable parameter tensors / 2,748,890 elements;
- shadow all 26 persistent floating buffers / 3,616 elements using the same charged-time decay;
- copy all 13 integer buffers / 13 elements from the latest sampled state;
- initialize every shadow from the first due online state after progress 0.75;
- keep `previous_parameters` only for consecutive-sample distance audit;
- keep all shadows detached, outside optimizer/SAM ownership, on the model device;
- update after the sole optimizer step and SAM exact restore, before the existing CUDA synchronization so cost is charged.

BatchNorm `running_mean` and `running_var` use the shorter EMA. Convex interpolation preserves nonnegative running variances when online/shadow values are valid. `num_batches_tracked` remains latest-sample copied, exactly as EXP-011. Do not use live BN buffers, parameter-only averaging, or an uncharged BN recalibration pass; each would create a different state policy.

Evaluation remains exactly once per epoch: live before the first EMA sample and EMA afterward. The same exception-safe no-grad swap must copy all EMA parameters and persistent buffers into the model, call frozen `Eval.evaluate` once, then restore online state, module modes, optimizer identities, and RNG-external state exactly. Never evaluate parent-horizon and candidate-horizon shadows in the same metric run.

## Compatibility with Cadence 31 and SAM

Cadence 31 remains coprime to period-two SAM. EXP-011 produced 80 ordinary and 80 SAM samples; the candidate should retain a parity difference of at most one despite small throughput variation. Every sampled state is post-optimizer and post-SAM-restore when applicable.

The shorter half-life changes only how sampled full states are combined. It does not alter:

- which steps use SAM;
- SAM radius, first/second loss, RNG replay, BN suppression, perturbation snapshots, or restore;
- Nesterov momentum, LR, weight decay, or drop path;
- CutMix gate/geometry/generators or independent-image exposure;
- online parameters, gradients, BN buffers, RNG state, or optimizer state.

Since sample cadence and foreach work are identical, charged overhead and memory should match EXP-011. The shorter coefficient is one different CPU scalar per sample, not a new GPU operation.

## Expected Lag and Plateau Effect

If EMA/live distance scales approximately with temporal mean age along the same local trajectory, reducing mean age by one third predicts final relative parameter distance near:

```text
0.015063 * (12.5 / 18.75) ~= 0.01004
```

This is a diagnostic expectation, not a correctness requirement. A final distance around 0.8-1.3% would show reduced lag without collapse to the live model. BN mean/variance distance and variance-ratio spread should also contract, but nonlinear trajectory curvature can violate simple scaling.

The hoped-for accuracy mechanism is that recent low-LR states and their matching BN statistics receive more mass, lifting EXP-011's 95.493 EMA plateau while retaining enough averaging to suppress live checkpoint variation. The opposing risk is that EXP-011's gain came from its longer smoothing horizon; shortening it may simply reintroduce live noise and lower the selected maximum.

The evidence does not identify the optimum. *How to Scale Your EMA* supports time/cadence-derived horizons, not 12.5 seconds specifically. SWA supports late averaging but often relies on a non-collapsing LR, unlike this cosine tail. EXP-011 is one fixed-seed package result and intentionally did not evaluate live tail checkpoints, so there is no direct evidence that lag rather than variance limits its plateau.

## Implementation Scope

Modify only `train.py`:

1. Change `EMA_TAIL_HALF_LIVES` from `4.0` to `6.0`, deriving `EMA_HALF_LIFE_S=12.5` from the unchanged tail duration.
2. Change the startup assertion/config output to require and print the new half-life.
3. Extend EMA audit output with continuous-time mean age, fixed-interval effective sample-size estimate, mass in the most recent 18.75 seconds, and a `horizon_mode=six_tail_half_lives` label.

Do not alter the EMA class's state coverage, cadence predicate, first-copy/update ordering, evaluation routing, swap/restore, online training, model, data, optimizer, CutMix, SAM, schedule, seed, timer, evaluator, or required summary keys. No new dependency, model tensor, forward, or evaluator call is added.

## Audits and Smokes

1. **Closed-form decay:** For fixed intervals including parent min/mean/max, compare production decay against `2^(-dt/12.5)` in FP64 and require the values above.
2. **Half-life identity:** Feed two states exactly 12.5 charged seconds apart and require 0.5 old-state retention; six successive half-lives must leave `1/64` initial weight.
3. **Partition invariance:** Drive the same piecewise-constant online state through different cadence partitions with equal elapsed time and require equivalent decay composition where state values are unchanged.
4. **Parent/candidate discrimination:** On one synthetic trajectory, run 18.75- and 12.5-second shadows in an accuracy-blind harness; require the shorter horizon to weight recent states more and differ from the parent while both remain finite. This parent shadow is harness-only and never evaluated in production.
5. **Full-state coverage:** Require exact 44/26/13 tensor counts and 2,748,890/3,616/13 element counts, correct float/int policy, no alias with model, optimizer, SAM, restore, or previous-state tensors.
6. **Cadence/SAM parity:** Simulate consecutive production steps and require first due sample at the same cadence-31 rule, no sample before 0.75, and ordinary/SAM count difference at most one.
7. **Online parity:** From matched fixed-seed state, run parent/candidate production steps and require bitwise-identical online model, BN, optimizer, RNG, losses, outputs, CutMix/SAM counters, and step timing decisions; only shadow values/config/audits may differ.
8. **Swap/restore:** Exercise successful and throwing fake evaluators and full GPU evaluation; require one evaluated candidate shadow, exact online/state/mode/optimizer/RNG restoration, and no second model metric.
9. **BN validity:** Require finite running means, positive finite running variances, finite ratio diagnostics, and no recalibration or live-buffer substitution.
10. **Distance audit:** Require all consecutive sampled-state distances finite and nonzero and final EMA/live distance finite. Record parameter and BN distances without making a post-hoc horizon choice.

## Feasibility

The candidate has the same number of state tensors, foreach operations, copies, distance computations, and evaluations as EXP-011. Expected memory is about 1,222 MiB and expected exposure is approximately 25,500-25,800 steps.

Use a paired physical-GPU-0 preflight with at least five alternating parent/candidate rounds covering the production 87.5% ordinary / 12.5% SAM weighted path, 160 cadence samples, and full swap/evaluate/restore. Require:

- paired median latency ratio `<=1.005` and p90 ratio `<=1.01`;
- projected steps `>=25,300`, while recognizing projection is only a feasibility estimate;
- realized metric-run dose gate preregistered at `>=25,200` steps;
- peak allocation `<1.30 GiB` and total-runtime projection `<600s`;
- parent passes the relative gate by construction and trial dispersion `<=5%`;
- zero online-parity, state, finite, RNG, coverage, cadence, swap, or restore failure.

Do not change cadence, move EMA work outside charged timing, or reduce audit work to rescue a failed gate.

## Mechanism Audit and Comparison

Preserve all EXP-011 audit lines and durably record exact values before transient log deletion. Compare candidate with these fixed parent references:

- 160 updates over 74.7736 seconds, 80/80 ordinary/SAM;
- interval mean 0.470274 seconds and decay mean 0.982765296;
- oldest coefficient 0.063025;
- consecutive parameter-distance mean 0.36667874;
- final EMA/live parameter relative distance 0.01506322;
- final BN mean/variance L2 `0.309745 / 0.337133`;
- final BN variance ratio range `0.963770-1.218576`;
- late-16 EMA range/mean/final `95.44-95.61 / 95.493125 / 95.46`.

Candidate expectations are decay mean around 0.97426, oldest coefficient around 0.0158, 145-165 updates with balanced parity, final relative parameter distance 0.8-1.3%, and contracted BN distance. These are mechanism diagnostics, not substitutes for accuracy and not grounds for a retry.

## Risks

- **Less smoothing:** Effective sample size falls from about 115 to 77 states, which may reintroduce checkpoint variance and erase EXP-011's gain.
- **Lag diagnosis may be wrong:** Parent final accuracy below best does not prove the EMA is stale; the online tail was intentionally not evaluated.
- **BN responsiveness can hurt:** Shorter buffer averaging may track noisy recent running statistics more closely, even if parameter lag improves.
- **Cosine-tail collapse:** Recent states may be highly correlated, so extra weight on them can add little while discarding useful earlier diversity.
- **Small expected effect:** A horizon-only refinement plausibly sits inside the known 0.14-0.29-point run noise and may not raise the stable plateau by the required amount.
- **Max-selected metric:** A new isolated best can pass while tail mean/final regress. Plateau context must accompany the formal result.
- **Scalar transfer:** Six half-lives is a preregistered charged-time choice, not a literature-proven optimum for this SAM/EMA composition.

## Testable Hypothesis and Falsification

EXP-011 is the 95.61 parent, so formal success requires `best_test_acc >=95.71%`. The proposal has low-to-medium confidence of clearing that threshold. The preregistered prediction is 95.62-95.76%, with supporting targets of late-16 EMA mean `>=95.58%` and final accuracy `>=95.55%`. A 95.71+ maximum without a higher plateau is formally valid but weak evidence for lag correction.

After preflight, run exactly once:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

Require physical GPU 0 to be the 97,871 MiB H20, exit 0, 299.5-301.0 charged seconds, total time below 600 seconds, at least 25,200 steps, unchanged 2,748,890 parameters, complete summary, one evaluation per epoch, unchanged CutMix/SAM and cadence-31 dose, exact full-state restore, finite nonzero shadow distances, valid 12.5-second decay audit, and zero coverage/nonfinite/RNG/restore failures.

The proposal is falsified by `best_test_acc <95.71%`, preflight/dose failure, any extra evaluator call, incorrect horizon arithmetic, state-policy mismatch, or hard-constraint violation. Do not rerun, inspect live accuracy, blend parent/candidate shadows, or tune half-life/start/cadence from the result.
