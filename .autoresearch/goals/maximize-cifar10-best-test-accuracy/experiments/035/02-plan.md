# Plan EXP-035: Fixed SiLU Throughout ResNet-20
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement the isolated activation substitution and prove semantics
- [x] Create `autoresearch/maximize-cifar10-best-test-accuracy-035` from the integration branch; replace exactly the three source-level `F.relu` calls in `train.py` with fixed `F.silu`, with no other tracked change.
- [x] Run compile, Ruff, format, pre-commit, scope, diff, and AST checks; prove the same 19 Conv/19 BN/one Linear modules, 1,073,962 parameters, state keys/order, optimizer membership, graph shapes, and exactly 19 dynamic SiLU sites with no model ReLU remaining.
- [x] Prove fixed beta-1 SiLU values/derivatives on CPU/CUDA, identical candidate/control initial tensors and RNG states, no forward RNG consumption, finite production-batch site/logit/loss/gradient statistics, and no initial candidate-only concentration.

### Milestone 2: Veto unsafe activation and optimizer geometry on immutable data
- [x] Re-hash and validate the registered EXP022 200-batch strong corpus and EXP028 64-batch weak corpus; preserve byte-identical inputs, targets, order, hard/soft coverage, and pre/post hashes.
- [x] Pass controller identity/known-array gate-math self-tests and two predeclared production-default control/control calibrations, then run independent accepted/candidate SGD replays on the same 200 strong and 64 weak batches while recording site, pooled-feature, class, logit, loss, gradient, update, momentum, parameter, and BN trajectories.
- [ ] Authorize timing only if all state is finite/complete, no candidate-only >95% class concentration occurs, all whole/per-layer/site/pooled ratios satisfy the registered catastrophic bounds, BN state is valid, and terminal strong/weak loss EMAs remain bounded.

### Milestone 3: Establish fixed-budget feasibility and execute once
- [ ] On one idle H20, run one conditioning process plus seven alternating fresh control/candidate timing pairs using a 40%/40%/20% strong-hard/strong-CutMix/weak-hard step mix; require the registered latency, variability, exposure, VRAM, and wall projections.
- [ ] Re-query the moving baseline, verify the exact three-call diff, one H20, no stale owned log, accepted data/optimizer/schedule/timer/evaluator contracts, then run seed 42 exactly once under the 595-second guard.
- [ ] Parse and verify the one production log: valid summary, 300-second training budget, total under 600 seconds, at most the accepted 19 unique once-per-epoch evaluations with a terminal look, and `best_test_acc >= moving baseline + 0.10` for improvement.

## Code Changes

- **`train.py`**: replace `F.relu` with `F.silu` at the BasicBlock conv1-BN activation, BasicBlock post-add activation, and ResNet stem conv-BN activation. These three source edits produce 19 fixed beta-1 SiLU executions while leaving module/state topology, residual ordering, initialization, and every training/evaluation policy unchanged.
- **Ignored preflight controller/artifacts**: create experiment-local comparison code and JSON/log outputs under `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/035/`. The controller may hook disposable copied models and import production code, but no diagnostic, hook, timing helper, or threshold enters tracked production code.

## Configuration Changes

- Activation: ReLU -> fixed beta-1 SiLU/Swish at all 19 dynamic activation sites, using `F.silu(..., inplace=False)` semantics.
- Initialization: unchanged accepted `init.kaiming_normal_(m.weight)` for Conv2d and Linear. No SiLU gain, fan-out, residual scale, beta, learned activation parameter, or site/stage exception.
- Unchanged: width-2 postactivation ResNet-20, Option-A shortcuts, GAP classifier, 1,073,962 parameters, FP32/default-TF32, batch128, N1/M7, p0.5 alpha-1 CutMix, 80% boundary, ordinary momentum0.9, all-parameter decay1e-4, weak-tail LR0.01-to-0.0001 cosine, seed42, eight persistent workers, timer, evaluator, and summary.

## Execution Environment

- Method: local ignored semantic/trajectory/timing controller, then conditional `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1` from the project root.
- Resources: exactly one idle NVIDIA H20 with approximately 97,871 MiB, existing CIFAR data/environment, and the registered EXP022/028 corpora. No package installation, remote job, W&B, compiler, precision, layout, or dependency change.
- Estimated runtime: 5-10 minutes for semantic/trajectory gates, 5-8 minutes for fresh paired timing, and about 335 seconds for production if authorized.
- Log output: ignored `preflight.log`, `preflight-report.json`, and `timing-silu.json` serialized and fsynced before assertions; production output only in root `run.log`, never `tee` or streamed in full.
- Tool skill: `/research-execute`; no submission platform skill is needed.

## Abort Criteria

- Stop before timing/production for any tracked-scope or three-call-diff mismatch; topology/state/parameter/RNG mismatch; incorrect SiLU oracle; corpus hash/schema/order mutation; nonfinite/incomplete state; candidate-only >95% class share; failed registered logit/update/per-layer/site/pooled/BN/loss-EMA trajectory bound.
- Stop before production if weighted candidate/control mean step time exceeds 1.02, any pair exceeds 1.04, timing CV exceeds 2%, candidate p95 exceeds 1.05x control mean, projected exposure is below 26,360 updates, memory exceeds 700 MiB or paired delta64 MiB, or projected wall reaches 540 seconds.
- During production terminate for fatal/nonfinite/resource/lifecycle/target/timer assertions, no progress for 120 seconds, or the 595-second process guard. Do not stop for low intermediate accuracy, switch fit, train loss, NLL, or a likely sub-threshold final result.
- One valid production completion only. No beta/site/activation variant, initializer/gain, LR/decay/warmup/clipping, BN setting, fusion/approximation, precision/layout, seed/corpus reroll, evaluator change, threshold relaxation, or post-veto rescue is allowed inside EXP035.

## Verification Protocol

### Verification Procedure

1. **Baseline/source/scope (30s):** run `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`, `git status --short`, `git diff -- train.py`, and `git diff --name-only`. Require the current 94.15% baseline at `7c1e7d8`, only `train.py` tracked, user-owned `data/` untouched, integration ancestry, and no stale `run.log` or renamed run-log variant.
2. **Static/semantic construction (180s):** run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `uv run ruff format --check train.py`, and `uv run pre-commit run --all-files`; then run the ignored controller's semantic stage. Require exactly three `F.silu` source sites/19 dynamic calls, no model ReLU, unchanged graph/state/parameters/optimizer membership, identical candidate/control initialization and post-construction CPU/CUDA RNG, no forward RNG consumption, and CPU/CUDA SiLU value/derivative oracle agreement. Hash the controller source into every report; test ratio/gate functions on known arrays and feed identical telemetry into both arms, requiring exact expected ratios and zero vetoes before any candidate data is interpreted.
3. **Initial production-batch function (180s):** on registered real hard/soft batches, hook disposable models at all 19 activation sites. Record site input/output/gradient RMS and signs, pooled-feature norm/sign balance, logits, loss, class shares, and BN state. Require finiteness, active sites, no candidate-only >95% share, and candidate/control initial logit/pooled/loss/global-gradient ratios within `[0.25,4.0]`; output parity is not expected.
4. **Corpus integrity (60s):** require EXP022 strong corpus file SHA-256 `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946` with 200 batches/94 hard/106 soft and EXP028 weak corpus SHA-256 `ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032` with 64 hard batches. Validate tensor hashes, ranks, shapes, order, and unchanged pre/post bytes; clone pinned/shared RNG states before CPU replay if used.
5. **Trajectory safety (600s):** run the ignored controller under production-default backend flags. Replay strong at LR0.1, then weak with accepted cosine values sampled across progress `[0.80,1.00]`; precede candidate comparison with two fixed control/control repeats. Require the identity/known-array tests to pass, both control repeats to serialize complete ordinary-variation evidence, and then require candidate finite/complete state, positive BN variances/exact 264 counters, zero candidate-only >95% class-share steps, candidate/control whole logit/gradient/update ratios <=5, whole update <=25% parameter norm and <=5x its preceding 16-step median, every trainable-tensor update <=50% norm, site-output/site-gradient/pooled-feature RMS ratios `[0.20,5.0]`, and terminal strong/weak loss EMA <=1.5x control. Thresholds are frozen before candidate replay; shared or control/control behavior is reported rather than erased. These are real registered post-transform production batches and a catastrophic-geometry screen, not a synthetic fit or accuracy proxy.
6. **Paired fixed-budget timing (900s):** after one unscored conditioning process, run seven alternating fresh control/candidate pairs on the idle H20 with production-default flags. Per arm warm 100 complete steps for each path, then measure 400 strong-hard, 400 strong-CutMix, and 200 weak-hard complete transfer/forward/loss/backward/SGD/synchronize steps. This exactly covers the production interval from `t0` through synchronization; loader transforms and iterator wait occur before `t0` and are unchanged by this activation-only diff. Require weighted mean ratio <=1.02, every pair <=1.04, both trial-mean CVs <=2%, candidate p95 <=1.05x control mean, projected `floor(26,898 * control_mean / candidate_mean) >= 26,360`, peak <700 MiB and delta<=64 MiB, finite state, and projected total wall<540s. Treat the historical projection as preflight only; the actual run's step floor remains load-bearing. Serialize trial order, path metrics, CUDA forward/backward/update components, clocks/utilization, and source hash before assertions.
7. **Production (595s):** re-query the baseline and verify one idle H20 with `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader`; verify exact diff and absence of stale owned logs, then run once with `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`. Require exit0, one finite ten-field summary, `training_seconds` in `[300.0,301.0)`, `total_seconds <600`, params1,073,962, one near-80% loader switch/eight workers stopped, 45-55% strong CutMix, hard weak targets, at least26,360 steps, no fatal signal, and at most19 unique evaluation epochs with at most one look per epoch plus a terminal look. The 19-look ceiling implements the EXP013 max-metric opportunity-bias finding; fewer looks from slower execution are conservative.
8. **Metric verdict (20s):** run `grep '^best_test_acc:\|^peak_vram_mb:' run.log` and parse the complete summary/evaluation lines. Re-query the index immediately before classification. All goal integrity conditions plus `best_test_acc >= queried baseline + 0.10` (currently >=94.25%) is improvement; a valid lower result is no-improvement; any preflight veto is invalid with NaN. Record switch accuracy against89.73, first weak against93.16, final NLL against0.1934, site/pooled diagnostics, steps, eval count, runtime, and VRAM, but none may rescue or invalidate the primary verdict. A bare threshold pass is protocol-valid single-seed evidence, not a statistically causal claim.

### Informational Metrics (Optional)

- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, `num_params`: parse the ten-field tail of `run.log` only after a complete valid run.
- Strong switch/first weak/peak/final accuracy and NLL: parse the registered evaluation and augmentation-switch lines; compare to EXP010's 89.73/93.16/94.15/94.15/0.1934 anchors.
- Activation mechanism: use `preflight-report.json` for site sign/RMS/gradient distributions, pooled-feature cancellation, class shares, trajectory ratios, BN state, and corpus/report hashes.
- Fixed-budget cost: use `timing-silu.json` plus production summary for control/candidate path times, weighted ratio, projected/actual exposure, CV/p95, CUDA stage breakdown, and VRAM.

## Adversarial Review Response

- Added controller identity telemetry, known-array ratio/gate tests, controller source hashes, and two control/control repeats before candidate interpretation.
- Retained trajectory vetoes because the registered corpora are real post-transform production batches and prior candidate-only one-class states are catastrophic process-integrity evidence; a veto remains NaN rather than a claimed accuracy loss.
- Removed the arbitrary lower evaluation-count bound but retained the accepted 19-look ceiling to prevent extra max-metric opportunities under the EXP013 project insight.
- Clarified that persisted timing matches the production counted interval and that actual production exposure/wall checks, not the historical projection alone, determine validity.
- Clarified 40%/40%/20% path weights versus 400/400/200 measured steps.
