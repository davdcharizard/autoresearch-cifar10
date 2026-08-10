# Plan EXP-038: Output-RMS-Matched Cosine Classifier
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement and prove the isolated angular head
- [x] Create `autoresearch/maximize-cifar10-best-test-accuracy-038` from the integration branch; change only tracked `train.py` with the frozen cosine head and preserve the accepted evaluator cadence exactly.
- [x] Pass compile, Ruff, format, pre-commit, diff/scope checks, and prove identical construction state/RNG, 19 Conv/19 BN/one Linear, 1,073,962 parameters, and unchanged accepted backbone/pooling/loaders/optimizer.
- [x] Match FP64 forward/input/weight-gradient oracles on random, tiny, and zero vectors; prove finite epsilon behavior, lifetime logit bound, unused bias state, and reproduce the frozen accepted/cosine RMS calibration on the hashed EXP022 batches.

### Milestone 2: Qualify short safety, slow norm drift, and fixed-budget cost
- [x] Re-hash registered EXP022/028 corpora and pass two accepted-control calibrations before a 200-strong/64-weak candidate replay; require finite complete state, bounded logit/update geometry, exact BN/target semantics, and no candidate-specific concentration.
- [ ] Cycle the immutable corpora through a separate 8,192-strong/2,048-weak survival replay. Not run: the candidate already exceeded the authoritative lifetime row max/min bound by step4 and at 261/264 short-replay looks, so the long stage could not pass without changing the plan after results.
- [ ] Run one H20 conditioner and seven counterbalanced fresh-process timing pairs over complete hard/CutMix/weak steps; require stable non-catastrophic overhead, bounded memory, and total-wall feasibility.

### Milestone 3: Execute and verify exactly once
- [ ] Re-query the moving baseline, confirm only the reviewed `train.py` diff, one idle H20, accepted once-per-epoch evaluation behavior with at most 19 looks, and no stale completed log.
- [ ] Run seed42 once under `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`; do not reroll, rescue, or terminate for low finite metrics.
- [ ] Parse the complete summary and trajectory; require protocol integrity and `best_test_acc >= 94.25%` for improvement, while using switch fit and norm dynamics to judge whether the hypothesized mechanism survived.

## Code Changes

- **`train.py`**: add frozen `COSINE_SCALE = 22.786916732788086`; after accepted global-average pooling and flattening, L2-normalize pooled features and `self.fc.weight` with `eps=1e-6`, then return `COSINE_SCALE * F.linear(features, weights)` without functional bias. Keep the existing `nn.Linear(128,10)` construction and parameter ordering so initialization, state, parameter count, and RNG are unchanged; its bias remains stored but receives no gradient/update. This is the only tracked behavioral change.
- **Ignored EXP038 controllers/artifacts**: construction/formula/calibration, immutable replay, drift, timing, and log-parsing helpers plus durable JSON/log evidence. None enter production or git.

## Configuration Changes

- Classifier function: affine `W h + b` -> `22.786916732788086 * normalize(W) @ normalize(h)`, `eps=1e-6`; the literal comes from accepted output-RMS matching on the first pre-existing EXP022 strong batch, not labels, accuracy, or a sweep.
- Unchanged: width2 postactivation ResNet20, global average pooling, batch128, seed42, FP32/default TF32, N1/M7 and alpha1 CutMix probability0.5 through80%, hard weak tail, ordinary SGD momentum0.9, all-parameter decay1e-4, `0.1` hold and `0.01→1e-4` cosine, workers, timer, accepted `(0.2,0.4,0.6,0.7)` plus per-epoch weak-tail evaluator cadence, summary, and 1,073,962 stored parameters.

## Execution Environment

- Method: local ignored controllers followed conditionally by `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`.
- Resources: exactly one idle NVIDIA H20 with 97,871 MiB; existing environment, CIFAR data, and immutable corpora; no dependency, remote job, or W&B changes.
- Estimated runtime: 10-16 minutes construction/short+long safety, 5-8 minutes paired timing, and about 330-360 seconds for the sole production run if authorized.
- Log output: controller output to ignored experiment-local logs/JSON; production stdout/stderr only to root `run.log`, never `tee` or streamed wholesale.
- Tool skill: `/research-execute` for implementation, preflight, monitoring, and production execution.

## Abort Criteria

- Before production, abort and classify the candidate invalid for scope/source/calibration/oracle mismatch; stale or changed corpus; nonfinite state; logit-bound violation; candidate-specific persistent class concentration; failed accepted controls; unsafe observed row/feature norms or updates; sustained late row-norm decay; strong-fit proxy failure; unstable/catastrophic timing; changed/more-than-once-per-epoch evaluation behavior; non-idle/wrong GPU; or wall projection >=540 seconds.
- Short replay catastrophic bounds: no candidate update above25% of its parameter norm or5x its preceding16-step median; no qualified whole/classifier/backbone gradient/update/logit statistic above5x the accepted-control envelope; strong and weak terminal loss EMA ratio <=1.5; exact264 BN increments; candidate-only concentration means >95% one-class share for two consecutive or three total matched steps while qualified controls do not. For a scalar statistic, the accepted envelope is the closed min/max across both controls at the matched look, enlarged only by a preregistered absolute floor (`1e-8` norms, `1e-6` losses/logit RMS); candidate ratios use the larger absolute control magnitude and that floor. The controller self-tests these rules on known arrays before reading candidate results.
- Long replay effect-specific bounds: every classifier row norm >=0.50, row max/min <=3, batch-median pooled-feature norm >=1.0, no row update >50% of its pre-update norm, and terminal reciprocal-row-norm and classifier tangent-update RMS <=2x their first256-step medians. Across the final four non-overlapping1,024-step strong windows and final two1,024-step weak windows, no row may lose >10% norm in each consecutive window and the final window loss must be <=10%; this is an observed survival gate, not a production-horizon extrapolation. Final512 hard-view top-1 must be no more than10 points below the lower qualified control, and strong/weak loss EMA remains <=1.5x the control envelope. Generic whole-feature/logit divergence is informational only after trajectories separate.
- Timing aborts only for aggregate candidate/control mean step ratio >1.05, any pair >1.08, trial/rate CV >=3%, peak allocation >=650 MiB, wall/count >1.10, or projected total >=540 seconds. Ordinary sub-5% overhead is priced by the real fixed-time accuracy run.
- During production, stop only for fatal/nonfinite/resource/lifecycle/timer faults, 120 seconds without progress, or the 595-second guard—not low finite loss/accuracy or a weak switch checkpoint. No scale, epsilon, LR, decay, bias, phase, loss, or rerun rescue is allowed.

## Verification Protocol

### Verification Procedure

1. **Prerequisites, baseline, and scope (30s):** before branching or source edits, require both registered corpus files to exist and match EXP022 SHA `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946` and EXP028 SHA `ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032`; validate their tensor schemas and frozen calibration helper against known arrays. Run `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`; require `best_test_acc`, higher-is-better, 94.15 at `7c1e7d8`, hence threshold94.25. Check `git status --short --branch`, ancestry, `git diff -- train.py`, and `git diff --check`; preserve untracked `data/` and allow no other tracked change.
2. **Static and construction (180s):** run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `uv run ruff format --check train.py`, and `uv run pre-commit run --files train.py`. Run the ignored construction controller and require identical named state/RNG after construction, exact inventory/parameter count, one optimizer group, unchanged data/schedule, stored bias present, and bias `grad is None`/bitwise unchanged after updates.
3. **Formula and calibration (180s):** run the ignored FP64 oracle over ordinary/tiny/zero vectors; compare logits and VJPs to a separately coded formula, require finite clamp behavior and `max(abs(logit)) <= scale*(1+1e-6)`. Re-hash EXP022 and reproduce accepted RMS `2.7600300312042236`, unit-cosine RMS `0.12112344801425934`, scale ratio `22.786916732788086`, matched candidate RMS, and the second hard-batch near-match before candidate authority.
4. **Short trajectory (600s):** revalidate corpus bytes, target ranks/sums, and hashes before/after. Freeze and hash the controller source plus every threshold before any candidate arm. Run two accepted controls first, serialize/fsync evidence before assertions, then replay candidate over200 strong LR0.1 and64 weak cosine batches. Apply the short replay abort bounds with the explicit denominator-safe envelope definition above.
5. **Long drift and fit survival (1200s):** in fresh arms from identical seed42 state, cycle the unchanged registered batches deterministically for8,192 strong steps at LR0.1 then2,048 weak steps with LR mapped evenly across progress0.8→1.0 on the production cosine. One accepted control runs before the candidate; the two short controls have already qualified generic gates. Snapshot every64 steps; record all row/feature norms, reciprocal norms, radial/tangent classifier gradients/updates, momentum, logits, losses, class shares, BN state, and hard-view top-1. Require all observed effect-specific long-replay bounds and hash-identical input sequence; no extrapolated production-horizon claim is made.
6. **Timing (900s):** after one unscored device conditioner, run seven alternating fresh accepted/candidate pairs. Each uses identical initialized state and byte-identical registered tensors, 100 warmups, then at least1,000 complete synchronized steps at 40% hard strong/40% CutMix/20% weak weighting, including H2D, forward, CE, backward, SGD, and synchronization. Persist raw trials before assertions; apply timing abort bounds and verify the unchanged evaluator condition cannot call more than once per epoch or exceed the accepted19-look ceiling under measured candidate epoch timing.
7. **Production (595s):** re-run baseline/scope/GPU checks with `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader`; require one idle H20. Remove only stale root experiment logs, never `data/`, then execute once with the declared timeout/redirection. Monitor bounded progress/tail reads without `tee` and without a second run.
8. **Summary integrity (30s):** run `grep '^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:' run.log`. Require exit0, ten finite fields, training `[300,301)`, total `<600`, params1,073,962, one switch near80%, eight stopped workers, CutMix45-55%, hard weak targets, unchanged accepted evaluation logic, and at most19 unique once-per-epoch evaluations including terminal.
9. **Verdict (20s):** compare parsed `best_test_acc` to94.25. All user-defined goal conditions plus `>=94.25` is improvement even if an informational mechanism diagnostic misses; a complete lower metric is no-improvement; process/scope/integrity or pre-production safety veto is invalid/NaN; infrastructure failure with no trustworthy result is crash. Never reroll. Record switch accuracy against89.73 (>=89.0 supports, but does not determine, the mechanism), first-weak, best/final/NLL, exposure, row/feature/tangent drift, timing, and wall metrics.

### Informational Metrics (Optional)

- Final ten summary fields: parsed from `run.log` with the command above.
- Evaluation dynamics: all `eval ep` and `augmentation_switch` lines; record 80% switch, first weak, peak epoch, final, best-final gap, and final NLL.
- Angular-head dynamics: controller JSON row/feature norm quantiles, reciprocal norms, radial/tangent gradient/update fractions, logit bound utilization, class shares, loss EMA, consecutive-window drift, and stored-bias hash.
- Fixed-budget cost: seven-pair aggregate/per-pair step ratios, CV, memory, projected and actual steps/images, and total wall projection.

## Adversarial Review Response

- Extends the original264-step gate to a separate10,240-step observed survival replay with consecutive late-window norm checks, directly addressing slow inverse-norm amplification without trusting a linear full-horizon extrapolation.
- Makes pre-production strong-view fit control-relative; production switch fit remains attribution evidence only, so it cannot discard a run that passes every user-defined goal condition.
- Keeps the accepted global LR/decay and preserves the stored Linear/bias construction, while treating output-RMS parity as an initialization control rather than a Jacobian-equivalence claim.
- Preserves the accepted evaluation code exactly and enforces only its existing once-per-epoch and at-most19-look integrity ceiling, avoiding an evaluator-cadence confound.
