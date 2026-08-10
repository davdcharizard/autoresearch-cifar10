# Plan EXP-034: Conv2d-Only Kaiming Fan-Out Initialization
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement and prove exact initialization semantics
- [x] Create `autoresearch/maximize-cifar10-best-test-accuracy-034`; modify only `_weights_init` in tracked `train.py`, using explicit fan-out/ReLU for Conv2d and retaining the literal accepted Linear call.
- [x] Pass compile/Ruff/format/pre-commit/scope/AST checks and prove 19 Conv/19 BN/one Linear, 1,073,962 parameters, unchanged graph/shapes, identical post-construction RNG, 16 bitwise-equal Conv tensors, and exact analytic rescaling of only three unequal-fan Conv tensors.
- [x] On immutable production inputs, prove expected pre-BN scaling, near-invariant train-mode post-BN activations/logits/loss, finite state, and unchanged Linear/BN/bias/buffer values.

### Milestone 2: Veto unsafe optimizer geometry on byte-identical data
- [x] Re-hash and validate the existing 200-batch strong and 64-batch weak corpora; never regenerate, filter, or reorder them.
- [x] Replay independent accepted/candidate models with ordinary SGD and record loss/class/logit/gradient/update/parameter/momentum/BN trajectories plus named statistics for the stem and two widening convolutions.
- [ ] Authorize production only with no candidate-only class concentration, finite/complete state, bounded whole-model and per-layer updates/logits, positive BN variance, exact counters, and bounded terminal strong/weak loss EMA.

### Milestone 3: Execute and verify exactly once
- [ ] Confirm the moving baseline, exact initializer-only diff, no stale log, one idle H20, and unchanged data/optimizer/schedule/timer/evaluator contracts; then run seed 42 once under the 595-second guard.
- [ ] Require a complete fixed-budget run with exactly the baseline's 19 unique terminal-inclusive evaluations and `best_test_acc >= moving_baseline+0.10`; never rerun a valid result.
- [ ] Record switch/first-weak/peak/final/NLL/exposure/runtime/VRAM plus initialization and safety evidence; diagnostics cannot override the formal metric.

## Code Changes
- **`train.py`**: split `_weights_init`: `nn.Conv2d` receives `init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")`; `nn.Linear` retains `init.kaiming_normal_(m.weight)` verbatim. No other production line changes.
- **Ignored preflight controller**: create `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/034/preflight_fanout.py` and reports/logs outside git to compare accepted/candidate constructors and replay the fixed corpora. It may instrument copied models only; no diagnostic hook or counter remains in production.

## Configuration Changes
- Conv initialization mode: PyTorch default `fan_in` -> explicit `fan_out` with ReLU nonlinearity. Linear remains default fan-in.
- Only `conv1` `[32,3,3,3]`, `layer2.0.conv1` `[64,32,3,3]`, and `layer3.0.conv1` `[128,64,3,3]` change, at registered candidate/control scales `sqrt(3/32)=0.306186`, `1/sqrt(2)=0.707107`, and `1/sqrt(2)=0.707107`.
- Unchanged: width-2 postactivation ResNet-20, 1,073,962 params, BN defaults, FC bias, graph, FP32/default-TF32, batch128, N1/M7, p0.5 alpha-1 CutMix, 80% boundary, LR/momentum/all-parameter decay, weak tail, seed42, workers, timer, and evaluator.

## Execution Environment
- Method: ignored exact-construction/trajectory controller, then conditional local `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`.
- Resources: one idle 97,871-MiB H20 and existing environment/data. Trajectory subprocesses retain and record the same default backend flags as production; they do not enable deterministic algorithms or alter TF32/cuDNN settings.
- Estimated runtime: semantic/trajectory preflight 2-6 minutes; production about 335 seconds if authorized. No paired timing campaign because initialization adds no recurring operator, shape, or memory traffic.
- Log output: `preflight.log` plus fsynced JSON under ignored `experiments/034/`; production only `run.log`, with no `tee` or production diagnostics.
- Tool skill: `/research-execute`; no remote job or W&B.

## Abort Criteria
- Stop before production on scope/AST/parameter/graph/RNG/tensor-scale/Linear/BN mismatch; corpus hash/schema/order mutation; nonfinite or incomplete model/optimizer/BN state; candidate-only >95% class share; registered logit/update/per-layer/EMA bound failure.
- During production stop for fatal/nonfinite/resource/lifecycle/target/timer assertions, no progress for 120 seconds, or the 595-second guard. Do not stop for low intermediate accuracy, switch fit, loss, or NLL.
- One valid completion only; at most one controller retry for a documented mechanical/infra fault that preserves seed, corpora, thresholds, and candidate semantics. No stem exclusion, transition-only variant, scale interpolation, fan-out Linear, BN epsilon, LR/decay/warmup/clipping, seed/corpus reroll, evaluator change, or gate relaxation.

## Verification Protocol

### Verification Procedure
1. **Baseline/source (10s):** run `exp-index.sh baseline` on `04-results.tsv`, derive the threshold as moving baseline+0.10 (currently 94.25 from 94.15 at `7c1e7d8`), require integration ancestry, only user-owned `data/` untracked, no stale `run.log`, and pristine baseline `train.py` before branching.
2. **Static and exact construction (180s):** compile/Ruff/format/pre-commit/diff/AST; run the ignored controller's construction stage. Require 19 Conv/19 BN/one Linear, 1,073,962 params, identical module/state ordering and post-construction CPU/CUDA RNG, bitwise equality for every unaffected tensor/buffer, and changed tensors equal accepted times analytic `c` at `atol=1e-7, rtol=1e-6`. Serialize before assertions.
3. **Initial function (180s):** on real immutable hard and soft batches, hook only disposable copied models. Require pre-BN RMS ratios within 2% of analytic scales, post-BN activation RMS ratios `[0.98,1.02]`, relative train-logit L2 <=0.02, loss ratio `[0.95,1.05]`, finite state, positive BN variances, and exact counters. Eval-mode differences are recorded, not gated.
4. **Corpus integrity (60s):** require EXP022 strong corpus SHA-256 `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946` with 200 accepted post-policy batches and EXP028 weak corpus SHA-256 `ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032` with 64 hard batches; validate every tensor/target rank/hash and preserve bytes before/after.
5. **Trajectory safety (600s):** `timeout 600s uv run .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/034/preflight_fanout.py > .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/034/preflight.log 2>&1`. Replay strong at LR0.1, then map the 64 weak records monotonically across progress `[0.80,1.00]` and apply the production cosine from0.01 to0.0001 in fresh processes with production-default backend/TF32 flags. Require identical accepted/candidate backend flags; no candidate-only >95% class share; all state finite/complete; candidate whole-model update <=25% of parameter norm, <=5x control and <=5x preceding 16-step median; each changed-Conv update <=50% of its norm; per-step candidate/control logit-RMS/raw-gradient/update ratios <=5; positive BN variances/exact 264 counters; terminal strong/weak loss EMA <=1.5x control; fsynced report before assertions. This is a catastrophic-geometry screen, not evidence against the slow strong-phase underfit seen in EXP012/015; only production measures that failure mode.
6. **Production (595s):** re-query baseline, confirm exact initializer-only diff and one idle H20, then remove only a stale owned `run.log` if present and run the guarded command once. Require exit0, one finite ten-field summary, training `[300.0,301.0)`, total<595, params1,073,962, one near-80% switch with eight workers stopped, 45-55% CutMix, hard weak targets, no fatal signal. Steps/VRAM are consistency diagnostics because the runtime graph is unchanged, not post-result validity filters.
7. **Evaluator/verdict (20s):** query the indexed baseline and EXP010 reference before parsing; require exactly the baseline's 19 evaluation lines on 19 unique epochs, at most one per epoch, and one terminal evaluation. Integrity plus `best_test_acc >= queried_moving_baseline+0.10` is improvement; a valid lower result is no-improvement. Compare switch to 89.73/87.08, first weak to 93.16, NLL to0.1934, steps to26,898, VRAM to598.7MiB, and best-final gap; none can rescue or invalidate the primary verdict. A bare +0.10 pass is formal under the user-approved single-seed protocol but remains noise-scale evidence and must not be described as statistically causal.

### Informational Metrics (Optional)
- Construction: changed tensor std/norm/scales, post-construction RNG hashes, initial pre/post-BN and train/eval logit differences.
- Safety: per-step class shares, loss/logit/gradient/update ratios, named-Conv relative updates, BN trajectories, strong/weak terminal EMAs, corpus/report hashes.
- Production: best/final top-1, final NLL, switch and first-weak accuracy, training/startup/total seconds, epochs/steps/params, CutMix rate, evaluation count, and peak VRAM copied into `03-execute.md` before log removal.

## Adversarial Review Response
- Retained the user-approved one-seed moving-baseline protocol and did not add post-hoc baseline reruns. The analysis must explicitly limit causal interpretation of a narrow pass.
- Clarified that the 264-step replay is only a catastrophic geometry/implementation screen; it cannot clear the known slow-horizon underfit risk, which production must measure.
- Aligned replay with production-default backend flags and replaced constant weak LR with the registered cosine sampled across the full weak-tail progress interval. The strong replay remains LR0.1, exactly matching the accepted 80% plateau.
- Tightened evaluator integrity from 18-19 looks to exact parity with EXP010's 19 because this initializer-only diff cannot legitimately change recurring runtime.
- Made the queried moving baseline load-bearing at verdict time and standardized the controller's full ignored path.
