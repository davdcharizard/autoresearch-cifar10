# Plan EXP-036: Reflection-Padded Strong and Weak Crops
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement and prove the isolated border-policy change
- [x] Create `autoresearch/maximize-cifar10-best-test-accuracy-036`; add only `padding_mode="reflect"` to the two accepted `RandomCrop(32, padding=4)` constructors in tracked `train.py`.
- [x] Pass compile/Ruff/format/pre-commit/scope/AST checks and prove both transform pipelines, crop geometry/order, model, data/target, optimizer, schedule, timer, evaluator, summary, and RNG draw count are otherwise unchanged.
- [x] Exhaust all 81 crop offsets on preregistered CIFAR images; require center crops equal, noncenter differences confined to padded source positions, expected shape/dtype/labels, and matched incoming/outgoing RNG for constant/reflection weak and strong transforms.

### Milestone 2: Qualify paired data semantics, safety, and loader capacity
- [x] Persist a paired source-index/per-sample-RNG corpus with aligned crop/flip/RandAugment/CutMix decisions and bitwise-equal targets: 32 strong batches (16 hard/16 CutMix) plus 16 weak hard batches; serialize source/input/target hashes before assertions.
- [x] Run denominator-safe identity/gate tests and two accepted control/control trajectory calibrations first. Only if both controls pass the frozen global bounds, replay identical-initialized accepted/candidate models on their corresponding paired tensors and require finite complete state with no candidate-only class concentration or gross whole-model excursion.
- [ ] Run three alternating fresh-process loader pairs, separately for strong and weak pipelines, over two post-warmup epochs per arm; require sustained rate, queue margin, non-rollover wait, lifecycle, and conservative wall projections before production.

### Milestone 3: Execute once and verify the metric
- [ ] Re-query the moving baseline, verify the exact two-keyword diff, one idle H20, no stale owned log, and unchanged model/optimizer/schedule/timer/evaluator contracts; then run seed 42 once under the 595-second guard.
- [ ] Require exit zero, a complete finite summary, 300 counted seconds, total below 600 seconds, one clean 80% loader switch, accepted CutMix/target semantics, and at most 19 once-per-epoch terminal-inclusive evaluations; record optimizer exposure as a consistency metric rather than a host-loader gate.
- [ ] Classify improvement only for all integrity conditions plus `best_test_acc >= moving baseline + 0.10`; record switch, first-weak, peak/final/NLL, steps, runtime, loader evidence, and border geometry without rerunning.

## Code Changes

- **`train.py`**: add `padding_mode="reflect"` to the weak and strong `transforms.RandomCrop(32, padding=4)` calls. No transform reordering, fill, phase-specific mode, tensor/v2 conversion, or any other tracked line changes.
- **Ignored experiment artifacts**: create paired-corpus, semantic/safety, and loader-timing controllers/reports only under `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/036/`. They may compare disposable models/loaders but no instrumentation enters production.

## Configuration Changes

- RandomCrop padding mode in both training phases: implicit `constant` -> explicit `reflect`.
- Frozen crop geometry: size32, padding4, offsets0-8, horizontal flip unchanged; strong-only N1/M7 and p0.5 alpha-1 CutMix remain in the accepted order/lifetime.
- Unchanged: eval transforms, width-2 ResNet-20, 1,073,962 params, initialization, FP32/default-TF32, batch128, seed42, forkserver/eight persistent workers, ordinary momentum0.9, all-parameter decay1e-4, 80% LR/data boundary, weak cosine0.01-to0.0001, timer, evaluator, and summary.

## Execution Environment

- Method: local ignored semantic/paired-corpus/trajectory and fresh-loader controllers, then conditional `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`.
- Resources: exactly one idle 97,871-MiB H20 for safety/production, existing CIFAR data and environment, CPU/forkserver workers for transform generation and loader timing; no new dependency, remote job, W&B, or package install.
- Estimated runtime: 2-4 minutes for paired semantics/safety, 2-4 minutes for fresh loader timing, and about 335-380 seconds production if authorized.
- Log output: ignored `preflight.log`, `preflight-report.json`, `paired-corpus.pt`, `loader-timing.log`, and `loader-timing.json`, each serialized/fsynced before gate assertions; production only in root `run.log`, never `tee`.
- Tool skill: `/research-execute`; no platform submission skill.

## Abort Criteria

- Stop before candidate replay if scope/transform/RNG/offset/target/corpus checks fail or either accepted control/control calibration violates the prospective global gate. Control failure invalidates the measurement protocol before candidate authority; do not tune a threshold and continue.
- Stop before production for nonfinite/incomplete state; candidate-only >95% class share on two consecutive or at least three total steps; candidate/control whole logit/gradient/update ratio >5; whole update >25% parameter norm or >5x preceding median; nonpositive BN variance/counter mismatch; or terminal strong/weak loss EMA >1.5x control. No per-site or per-zero-norm tensor ratio is allowed.
- Stop before production if either reflection loader sustains <95% paired batch rate or <1.25x contemporaneous training demand, candidate non-rollover p95 wait exceeds 1.5x paired control, workers fail lifecycle checks, projected wall reaches 540 seconds, or full source/target/RNG semantics diverge.
- During production terminate on fatal/nonfinite/resource/lifecycle/target/timer assertions, no progress for 120 seconds, or the 595-second guard. Do not stop for low intermediate accuracy/loss/NLL.
- One valid run only. No phase-only reflection, padding width/mode, worker count, transform order/device, seed/corpus reroll, threshold relaxation, or evaluator change inside EXP036.

## Verification Protocol

### Verification Procedure

1. **Baseline/source/scope (30s):** query `exp-index.sh baseline` (currently 94.15 at `7c1e7d8`), inspect `git status --short`, `git diff --name-only`, and `git diff -- train.py`; require only two `padding_mode="reflect"` additions in tracked `train.py`, user-owned `data/` untouched, integration ancestry, and no stale `run.log` variant.
2. **Static/constructor semantics (180s):** run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `uv run ruff format --check train.py`, and `uv run pre-commit run --all-files`; inspect AST/transform reprs. Require exactly two reflected RandomCrops and literal parity for every other transform, model, optimizer, schedule, timer, evaluator, and summary node.
3. **Offset/RNG differential (180s):** on preregistered real CIFAR images, enumerate all 81 offsets before flip/RandAugment/CutMix. Require exact center equality; noncenter differences only at crop pixels sourced from padding; matched output shapes/dtypes/labels; reflection semantics against a direct tensor oracle; and identical outgoing CPU RNG after matched weak/strong transforms. Record the finite-corpus changed area against the analytic 13.41% expectation without gating to that exact mean.
4. **Paired corpus (300s):** preregister source indices/order and clone contiguous parent-owned per-sample RNG states. Generate accepted/reflection views from identical incoming states, require identical outgoing states and aligned crop/flip/RandAugment metadata, then apply identical saved CutMix state and require bitwise-equal targets/permutations/boxes/outgoing state. Persist 16 hard+16 CutMix strong and 16 hard weak paired batches; hash files/tensors/metadata before and after replay. Never regenerate/filter after model behavior. This parent-process corpus is a bounded counterfactual semantics/safety proxy; it does not claim to reproduce the exact eight-worker production stream.
5. **Control-qualified global safety (600s):** self-test all gate math on known/identity arrays; run two accepted/accepted paired trajectories first under production-default CUDA. Each must have no one-sided concentration meeting the prospective failure definition (two consecutive or >=3 total >95% steps), all whole logit/gradient/update ratios <=5, whole update <=25% parameter norm and <=5x preceding median, positive BN variances/exact48 counters, and terminal phase EMA <=1.5x. Abort before candidate if controls fail. Then replay accepted/reflection arms from identical state on corresponding paired tensors using LR0.1 strong and accepted cosine samples over weak progress `[0.80,1.00]`; require the same frozen bounds, with no per-site/per-tensor ratios. Lower loss cannot waive a veto; a single isolated class transient is recorded, not fatal.
6. **Fresh paired loader capacity (600s):** run three alternating fresh-process pairs per strong/weak pipeline using exact production DataLoader settings. After worker/cache warmup, measure two full epochs per arm, separating first-batch and iterator-rollover waits. Require candidate sustained rate >=95% control and >=1.25x measured training demand, candidate non-rollover p95 <=1.5x control, no repeated starvation/worker failure, exact batch/target contracts, clean shutdown, trial CV<=5%, and conservative strong/weak-weighted total wall<540s. Do not use EXP033's invalid absolute rollover-inclusive p95 gate. Because `t0` begins after batch retrieval and GPU code is identical, host wait affects total wall rather than counted-step work; step count is only a production consistency metric.
7. **Production (595s):** re-query baseline; require exactly one idle H20 from `nvidia-smi`, exact diff, clean owned logs, and passed artifacts, then run once with `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`. Require exit0, one finite ten-field summary, training `[300.0,301.0)`, total<600, params1,073,962, one near-80% switch/eight workers stopped, strong CutMix45-55%, hard weak targets, no fatal signal, and <=19 unique at-most-once-per-epoch evaluations including terminal. Record steps against26,898 but do not invalidate a goal-compliant run on that informational comparison.
8. **Verdict (20s):** parse every EXP036 value from `run.log` using `grep '^best_test_acc:\|^peak_vram_mb:' run.log` plus the complete summary/evaluation/switch lines, then re-query the index. Integrity plus `best_test_acc >= queried baseline +0.10` (currently94.25) is improvement; a valid lower result is no-improvement; preflight veto is invalid/NaN. Compare the parsed EXP036 switch, first-weak, and final NLL to EXP010's 89.73/93.16/0.1934 anchors; the anchors are never transcribed as candidate results. Border geometry, loader rates, steps, and runtime are explanatory only. A bare pass or miss is noise-limited single-seed max-over-checkpoints protocol evidence, not a causal effect estimate, and never authorizes a confirmation reroll.

### Informational Metrics (Optional)

- Summary metrics: parse all ten final fields from `run.log` after a complete run.
- Accuracy dynamics: switch, first weak, best/final, final NLL, evaluation count, and best epoch from `run.log`, compared to EXP010 anchors.
- Border semantics: offset distribution, changed-pixel area, boundary values, paired metadata/RNG/target hashes from `preflight-report.json`.
- Loader feasibility: strong/weak sustained batches/s, demand margin, non-rollover median/p95, first/rollover waits, CV, worker lifecycle, and wall projection from `loader-timing.json`.

## Adversarial Review Response

- Made every candidate metric log-derived and labeled EXP010 values as comparison anchors only.
- Explicitly limited a bare pass/miss to noise-scale single-seed evidence with no reroll.
- Centered loader feasibility on total wall margin; removed optimizer-step count as a validity gate because waits occur outside `t0`.
- Required persistent/repeated candidate-only concentration for a data-policy veto and retained only denominator-safe whole-model statistics that accepted controls must pass first.
- Labeled the parent-process corpus a bounded proxy rather than the production worker trajectory.
- Reduced paired corpus and loader-trial scale while preserving offset/RNG/target semantics, control qualification, and wall projection.
