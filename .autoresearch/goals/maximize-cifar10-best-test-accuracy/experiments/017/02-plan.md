# Plan EXP-017: Learned Pool-First Transition Shortcuts
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Isolated production implementation
- [x] Confirm baseline 94.15 at `7c1e7d8`, clean accepted `train.py`, sole idle H20, and no stale run log.
- [x] Create `autoresearch/maximize-cifar10-best-test-accuracy-017` from the integration branch.
- [x] Add a `ShortcutConv` marker and replace exactly two Option-A shortcuts by `AvgPool2d(2,2) -> ShortcutConv(1x1,s1,bias=False) -> BatchNorm2d`; keep seven same-shape shortcuts as `Identity`.
- [x] Preserve accepted residual branches; isolate constructor draws in CPU RNG forks, initialize both projections sequentially from one generator derived from the active seed, and exclude them from the later model-wide initializer.
- [x] Pass syntax, diff-scope, exact parameter-count, topology, parameter-group, and accepted-contract checks.

### Milestone 2: Structural, RNG, and branch semantics
- [x] Create ignored controllers with explicit project-root import bootstrapping, then obtain external Claude source approval before running them; no fallback reviewer.
- [x] Prove exact pool/projection/BN settings, coordinate-ramp pooling semantics, branch shapes, and unchanged residual stride-2 paths.
- [x] Compare aligned seed-42 control/candidate construction: every shared tensor and post-construction CPU RNG state must be bitwise equal, and both projection tensors must match the expected sequential draws from the dedicated seed-derived generator.
- [x] Require exactly 1,084,586 parameters, exactly 10,624 new trainable parameters, two new BN states, and no stale pad path.
- [x] Verify hard and probability targets produce finite/nonzero new-parameter gradients and preserve all accepted shared behavior outside the changed shortcut sums.

### Milestone 3: Real-batch update and trajectory gate
- [x] Materialize a seed-fixed production N1/M7 stream with 45-55% CutMix and persist all failure evidence before raising.
- [x] On independent hard/soft first updates, require bounded shortcut/residual RMS, projection update norm, replay loss, concentration, BN state, and new-path recruitment.
- [x] Run paired accepted/candidate training on 200 distinct identical real batches; reject candidate-only concentration or loss divergence without tuning/rescue.

### Milestone 4: Paired H20 timing, exposure, and observation fairness
- [x] Run five alternating fresh-process control/candidate training pairs with 100 warmups and 500 measured exact production-region steps.
- [x] Require candidate/control ratio `<=1.0548`, at least 25,500 projected steps, stable p95/CV, and bounded memory.
- [x] Run five fresh inference pairs and project total wall below 540 seconds.
- [x] Project no more than 19 production evaluations; do not alter cadence to compensate for candidate epoch count.

### Milestone 5: Loader lifecycle
- [x] Measure 1,000 real strong batches, target provenance, iterator headroom, eight-worker shutdown, and a hard weak batch.

### Milestone 6: One fixed-seed production run
- [x] Reconfirm every conjunctive gate, baseline, scope, idle GPU, and absence of stale logs.
- [x] Run exactly once at seed 42 under the 600-second supervisor with output only in `run.log`.
- [x] Monitor bounded tails/process state; never early-stop or retry for unfavorable accuracy.

### Milestone 7: Integrity and metric verification
- [x] Parse ten finite summary fields and require the fixed timer, total wall, exact parameters, lifecycle, target, evaluation, and scope invariants.
- [x] Formal improvement requires `best_test_acc >=94.25`; 94.25-94.35 is reported as noise-consistent weak evidence, and candidate-mechanism support additionally requires at least 25,500 actual steps.
- [x] Record transition/switch/tail/NLL/exposure diagnostics and preserve `run.log` through analysis only.

## Code Changes
- **`train.py`**: introduce marker subclass `ShortcutConv(nn.Conv2d)` so the two new projections can retain their explicitly generated Kaiming weights while the accepted model-wide initializer remains byte-equivalent for shared Conv/Linear tensors.
- **`train.py` / `BasicBlock.__init__`**: for `stride==2` and changed channels, construct exactly `AvgPool2d(kernel_size=2,stride=2,padding=0,ceil_mode=False,count_include_pad=False)`, a bias-free stride-1 `1x1` `ShortcutConv`, and default `BatchNorm2d(out_channels)`; otherwise use `nn.Identity()`.
- **`train.py` / initialization**: create one CPU `torch.Generator` from `torch.initial_seed()` in `ResNet`, pass it through layer/block construction, isolate each `ShortcutConv` constructor inside `torch.random.fork_rng(devices=[])`, and Kaiming-initialize both projection weights sequentially with that generator. Return early for `ShortcutConv` in `_weights_init`. This preserves the accepted global stream/shared initialization, avoids reuse of a forked substream by later shared layers, and introduces no tunable secondary seed.
- **`train.py` / `BasicBlock.forward`**: replace conditional slice/pad code by `out += self.shortcut(x)`; do not alter either residual convolution, BN/ReLU ordering, or post-add ReLU.
- **Ignored experiment artifacts**: create `preflight_shortcut.py` and `timing_shortcut.py` under EXP017 with explicit project-root import bootstrapping and persistent failure JSON. They cannot modify production behavior or tracked files.

## Configuration Changes
- Transition shortcuts: raw `x[:, :, ::2, ::2]` plus zero channel padding -> pool-first learned normalized projection at `layer2[0]` and `layer3[0]` only.
- Parameter count: `1,073,962 -> 1,084,586` (+10,624: projections 2,048/8,192 and BN affine 128/256).
- Same-shape shortcuts: explicit `nn.Identity`, functionally unchanged.
- Unchanged: width 2; batch 128; FP32; seed 42; standard SGD; LR/momentum/all-parameter decay; N1/M7; p=0.5 alpha-1 CutMix; 80% phase switch; hard weak tail; timer; evaluator; checkpoints; workers.
- No blur on the residual path, strided projection, alternate pool, custom BN gamma, partial-identity initialization, shortcut-specific optimizer group, ECA, Nesterov, precision/layout change, or fallback shortcut.

## Execution Environment
- Method: local single-GPU commands from the project root.
- Resources: exactly one idle NVIDIA H20 with approximately 97,871 MiB, pinned by `CUDA_VISIBLE_DEVICES=0`.
- Estimated runtime: 4-8 minutes for reviewed preflights plus about 5.5-9 minutes for the sole production run; every command has a hard timeout.
- Log output: bounded preflight JSON/console summaries under EXP017; production output only to `run.log`, never `tee` or full streaming.
- Tool skill: none; local execution.

## Abort Criteria
- Preflight no-go on any scope/topology/RNG/shared-state/parameter/target/gradient/pooling-semantics failure; do not change the reviewed design.
- Abort if projected-shortcut RMS is outside `[0.25,4.0]` residual RMS; this is an implementation/catastrophic tripwire because both paths are normalized, not the primary scale safeguard. Record `>2.0x` as a diagnostic warning but do not tune or convert it into an undeclared veto; update norm, replay loss, and concentration are load-bearing.
- Abort if either projection first update exceeds 25% of its pre-update norm, replay loss exceeds 2x its own pre-update or paired-control replay, candidate-only class concentration exceeds 95%, BN state is invalid, or either new path is unrecruited.
- Abort if the 200-distinct-batch candidate has any non-finite state, candidate-only concentration above 95%, or terminal loss EMA above 1.5x control. Persist the failing step, targets, histograms, losses, and scale data before raising.
- Abort if training ratio exceeds 1.0548, projected steps fall below 25,500, candidate p95 exceeds 1.10x control p95, either trial-mean CV reaches 3%, peak allocation reaches 700 MiB or exceeds control by 96 MiB, inference ratio exceeds 1.08, inference CV reaches 3%, projected eval count exceeds 19, or projected total reaches 540 seconds.
- During production, terminate only for crash, non-finite output, missing progress beyond measured startup, GPU/resource/lifecycle/protocol fault, or 600-second timeout. Do not stop on low accuracy or switch underfit.
- A valid production run is any run that exits zero with the complete finite summary and all fixed scope/seed/timer/evaluator/lifecycle conditions, regardless of its accuracy. Never rerun such a run. Repair applies only when no usable valid summary exists and an independent controller/implementation/environment defect is demonstrated while exact pooling, projection, BN, initialization, and candidate scope remain unchanged. No fallback reviewer or experiment is allowed.

## Verification Protocol

### Verification Procedure

1. **Baseline, branch, scope, GPU (30 seconds).** Run the baseline query, `git status --short --branch`, `git rev-parse --short HEAD`, and `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader`. Require 94.15 at `7c1e7d8`, only known `data/`, and one idle 97,871 MiB H20. Remove only `run.log`; never clean `data/`.

2. **External implementation addendum.** After implementing production and controller sources but before executing any controller, give Claude the full production diff, plan, and controllers with the plan-critic prompt. Record successful output in `02-plan-review-implementation-addendum.md`. A non-zero/empty review is retried or treated as a credential blocker; never substitute self/subagent review. If source changes after review, obtain focused Claude approval before rerunning.

3. **Static/structural checks (60 seconds).** Run:
   ```bash
   git diff --check
   uv run python -m py_compile train.py
   git diff --name-only
   CUDA_VISIBLE_DEVICES=0 timeout 60s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/017/preflight_shortcut.py --structural
   ```
   Require `STRUCTURAL_GATE_PASS`, exact topology/shape/ramp/RNG/shared-state/gradient evidence in JSON, only `train.py` tracked, and parameter count 1,084,586. Inspect the full production diff.

4. **Real-batch safety (240 seconds).** Run:
   ```bash
   CUDA_VISIBLE_DEVICES=0 timeout 240s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/017/preflight_shortcut.py --numerical
   ```
   Require `NUMERICAL_GATE_PASS`, both hard/soft first-update gates, 45-55% CutMix across 200 distinct paired batches, and persisted JSON. Shortcut RMS above 2x is recorded for later attribution but remains a pass until the fixed 4x veto.

5. **Paired training/inference timing (480 seconds).** The reviewed controller uses one unscored conditioning process per benchmark group, five alternating fresh pairs, identical pinned workloads, unchanged backend/layout state, exact production timing boundaries, 100 warmups, and at least 500 measured training steps:
   ```bash
   CUDA_VISIBLE_DEVICES=0 timeout 480s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/017/timing_shortcut.py
   ```
   Require `TIMING_GATE_PASS`, raw trial JSON, ratio `<=1.0548`, all exposure/p95/CV/memory/inference/wall gates, at least 25,500 projected steps, and no more than 19 projected evaluations. Controller timing may not enable autotune, channels-last, compilation, precision changes, omit H2D/SGD/sync, or add candidate-only warmup.

6. **Loader/lifecycle (180 seconds).** Run:
   ```bash
   CUDA_VISIBLE_DEVICES=0 timeout 180s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/017/preflight_shortcut.py --loader
   ```
   Require 1,000 strong batches, CutMix `[45%,55%]`, median/p95 wait below 10%/20% of candidate step, exactly eight strong workers stopped before a hard weak batch, and no remaining process.

7. **One production run (600 seconds).** Reconfirm steps 1-6 and launch once:
   ```bash
   CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1
   ```
   Poll the existing process every 30-60 seconds and inspect only bounded tails/process state. Exit 124 is failure.

8. **Necessary-condition verification (60 seconds).** Extract the ten summary keys and `augmentation_switch`/`eval ep` lines with bounded `grep`. Parse numerically. Require finite summary, approximately 300 counted seconds as protocol integrity, total `<600`, parameters `1,084,586`, best accuracy `>=94.25`, one switch near 80%, eight stopped workers, CutMix `[45%,55%]`, hard weak targets, unique epochs, at most one eval per epoch, and at most 19 evaluations. Query the moving baseline again before verdict. A 94.25-94.35 pass is formal but explicitly noise-consistent weak evidence, not a precise causal estimate.

9. **Attribution and cleanup.** Mechanism support additionally requires actual steps `>=25,500`; an accuracy pass below that floor is accuracy-only and attribution-weak, and no amount of ordinary jitter authorizes rerun. Record shortcut RMS bands, switch/first-weak/final/best/NLL, strong/tail steps, eval count, BN state, memory, startup, and total wall versus EXP010. Keep `run.log` through analysis, then remove it. On no-go/no-improvement restore only `train.py`, return to integration, and preserve `data/`.

### Informational Metrics (Optional)
- Final accuracy/loss, training/total/startup seconds, VRAM, epochs, steps, and parameters: ten final `run.log` summary lines.
- Switch, first weak, tail slope, final/best gap, and evaluation count: bounded parse of `eval ep` lines.
- CutMix rate and lifecycle: `augmentation_switch` line.
- Shortcut/residual RMS, update ratios, trajectory safety, paired timing, memory, and projections: ignored EXP017 preflight JSON.
