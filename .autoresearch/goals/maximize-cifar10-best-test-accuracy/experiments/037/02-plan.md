# Plan EXP-037: Mean-Centered Stem Convolution
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement and prove the isolated projection
- [x] Create the EXP037 branch; add one `MeanCenteredConv2d` subclass and use it only for `ResNet.conv1` in tracked `train.py`.
- [x] Pass static/scope checks and prove identical construction RNG/state/parameters, 19 Conv/19 BN/one Linear, 1,073,962 params, one projected stem, and ordinary residual convolutions.
- [x] Match the decisive FP64 forward/input-gradient oracle; prove zero effective filter means, non-expansive raw/effective norms, and projected raw-weight gradients. The planned mechanism gate failed before a separate optimizer-recurrence controller was warranted.

### Milestone 2: Qualify mechanism, safety, and fixed-budget cost
- [x] On registered hard/CutMix views, require finite initial loss/post-BN/logit ratios and measurable post-BN/pooled divergence; repeat after a fixed short strong replay to veto a BN-washed-out mechanism. Failed at step64: candidate/control separation was only `1.61x` hard and `1.48x` CutMix versus the required `5x`.
- [ ] Re-hash EXP022/028 corpora, pass two accepted control calibrations, then replay candidate/control with denominator-safe global and stem-effective gates; require finite complete state and no persistent candidate-only concentration.
- [ ] Run one conditioner plus seven alternating fresh H20 timing pairs using the production subclass; require <=1.05 aggregate step ratio, stable trials, bounded memory, and wall projection<540s. Accuracy, not a 1% preflight cutoff, prices ordinary overhead.

### Milestone 3: Execute and verify once
- [ ] Re-query baseline, confirm exact stem-only diff/idle H20/no stale log, then run seed42 once under `timeout --kill-after=5s 595s` with output only to `run.log`.
- [ ] Require complete finite summary, 300-second budget, total<600, clean 80% switch/targets/CutMix, and <=19 terminal-inclusive once-per-epoch evaluations.
- [ ] Improvement requires all integrity gates and `best_test_acc >= moving baseline+0.10`; record actual switch/first-weak/peak/final/NLL/exposure/runtime without reroll.

## Code Changes

- **`train.py`**: define `MeanCenteredConv2d(nn.Conv2d)` whose forward uses `weight - weight.mean((1,2,3), keepdim=True)` with `_conv_forward`; instantiate only the stem with this subclass. No variance division, epsilon, cache, hook, parameter mutation, or other tracked change.
- **Ignored artifacts**: experiment-local construction/mechanism/trajectory/timing controllers and JSON/log reports only; production remains diagnostic-free.

## Configuration Changes

- Stem effective weight: raw Kaiming weight -> per-output-filter mean-centered weight every forward.
- Unchanged: stored initialization, BN, graph/shapes, constant crop padding, N1/M7+CutMix, width2 ResNet20, ordinary SGD/decay/LR, batch128, seed42, FP32/default-TF32, workers, timer, evaluator, summary.

## Execution Environment

- Method: ignored local controllers, then conditional `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`.
- Resources: one idle 97,871-MiB H20, existing environment/data/registered corpora; no remote job/dependency/W&B.
- Estimated runtime: 5-8 minutes preflight, 5-8 minutes timing, ~335 seconds production if authorized.
- Log output: ignored `preflight.log`/`preflight-report.json`/`timing.log`/`timing.json`; production root `run.log`, never tee.
- Tool skill: `/research-execute`.

## Abort Criteria

- Abort for scope/topology/RNG/oracle/projection/momentum mismatch; nonfinite state; zero mechanism survival; candidate-only persistent concentration; global logit/gradient/update >5x; whole update>25% norm or>5x median; stem effective update>25%; invalid BN/counters; phase EMA>1.5x.
- Mechanism survival requires post-BN RMS relative difference or pooled/logit relative L2 to exceed `max(1e-4, 5x the matching accepted control/control divergence)` at initialization and after64 strong steps. Controls run first; this detects a functional null, not accuracy direction.
- Abort timing only for aggregate ratio>1.05, any pair>1.08, CV>3%, peak>650MiB, wall/count>1.10, or projected total>=540s. Record projected steps but do not veto an otherwise feasible fixed-budget accuracy tradeoff.
- During production stop only for fatal/nonfinite/resource/lifecycle/timer faults, 120s no progress, or guard timeout—not low metrics. No all-layer/scaled/phase/init-only/decay/LR rescue or reroll.

## Verification Protocol

### Verification Procedure

1. **Baseline/scope (30s):** query index (94.15 at `7c1e7d8` currently), inspect status/diff/ancestry/logs; only reviewed `train.py` stem projection may differ and `data/` stays untouched.
2. **Static/construction (180s):** compile, Ruff, format, pre-commit, AST, inventory/state/RNG/optimizer checks; exact one subclass instance and unchanged 1,073,962 params.
3. **Projection oracle (180s):** compare FP64 `F.conv2d(x,W-W.mean)` outputs/gradients; require effective means<=1e-7, non-expansion, stored-state/RNG equality, and exact projected-data-gradient plus coupled-decay/momentum recurrence.
4. **Mechanism survival (300s):** hash/self-test the controller and production helper; require known-array projection/ratio math and identical accepted telemetry to pass. On real registered hard/CutMix views, hook pre/post-BN, pooled features, logits/loss at init and after64 fixed strong steps. Two accepted controls first establish matching divergence noise. Candidate must be finite, loss/post-BN ratios `[0.8,1.2]`, logit ratio<2, no persistent candidate-only concentration, and exceed `max(1e-4,5x control divergence)` in post-BN RMS or pooled/logit relative L2 at both looks.
5. **Corpus/global safety (600s):** verify registered EXP022/028 hashes/schemas; two accepted control pairs must pass frozen whole-model gates before candidate. Replay 200 strong LR0.1+64 weak cosine batches; require finite/complete state, exact264 BN counters, no persistent candidate-only >95% share, global ratios<=5, whole update<=25% and<=5x median, stem effective update<=25%, positive BN variance, phase EMA<=1.5. No zero-denominator/per-site gates; serialize controller/source/corpus hashes before assertions.
6. **Timing (900s):** conditioner plus seven alternating fresh pairs, each100 warmup+1000 measured real hard/soft/weak complete steps bound to the production subclass. Require aggregate<=1.05, pair<=1.08, CV<3%, peak<650MiB, no starvation, wall/count<=1.10, total<540s; serialize before assertions. Projected steps are informational and ordinary overhead is judged by the primary metric.
7. **Production (595s):** re-query baseline and H20/diff/logs, run once. Require exit0, finite ten fields, training `[300,301)`, total<600, params1,073,962, one ~80% switch/eight workers, hard weak targets, CutMix45-55%, <=19 unique once-per-epoch evaluations with terminal. Record steps/look count but impose no lower floor.
8. **Verdict (20s):** parse every EXP037 value from `run.log`; integrity plus `best_test_acc>=baseline+0.10` (currently94.25) is improvement, valid lower is no-improvement, veto is invalid/NaN. Compare parsed metrics to EXP010 anchors only; a bare pass/miss is single-seed evidence and never rerolled.

### Informational Metrics (Optional)

- Final ten fields and evaluation dynamics from `run.log`.
- Stem raw/effective/null norms, removed energy, projected gradient fraction, post-BN/pooled/logit divergence, global trajectory, BN and corpus hashes from `preflight-report.json`.
- Paired step/CUDA-stage ratios, CV, exposure, memory and wall projection from `timing.json`.

## Adversarial Review Response

- Acknowledges single-seed attribution limits without changing the user-approved verdict rule.
- Control-qualifies every candidate statistic and binds controller math/source/identity checks before authority.
- Uses a control-relative mechanism-survival floor to detect BN-washed-out nulls prospectively.
- Lets fixed-budget accuracy price ordinary projection overhead; timing vetoes only catastrophic feasibility.
- Removes production step/look lower bounds while retaining the 19-look anti-opportunity ceiling.
