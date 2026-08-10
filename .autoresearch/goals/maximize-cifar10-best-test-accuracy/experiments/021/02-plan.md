# Plan EXP-021: Deterministic Pool-First Option-A Shortcuts
- **Created**: 2026-08-06

## Goal and Registered Hypothesis

The moving baseline is `best_test_acc=94.15%` at `7c1e7d8`; a formal improvement requires `>=94.25%`. Replace only the two transition shortcut even-phase samples with deterministic nonoverlapping 2x2 average pooling before the accepted zero channel pad. The candidate should preserve at least 98% of EXP-010's 26,898 updates and improve transition information/generalization without adding parameters, learned state, randomness, data changes, or evaluation opportunities. The brainstorm's 94.30% value is retained only as a conditional upside point, not an evidence-weighted forecast: EXP-017 makes no-improvement the more likely outcome, with high mechanistic value either way. A bare 94.25% pass is protocol-valid but weak single-run causal evidence.

The negative-result discriminator requested by external idea review is registered before execution. Compare EXP-021 with EXP-017's 90.20% switch accuracy, 93.45% first-weak accuracy, 0.2024 final NLL, and 94.09% best. If deterministic pooling repeats the early-fit gains but again raises NLL above roughly 0.19 and misses 94.15%, attribute the late-generalization harm to pooling rather than EXP-017's learned projection/BN. Diagnostics never override the primary threshold.

Mandatory external Claude plan review completed with exit code 0 and is preserved in `02-plan-review.md`; no fallback reviewer was used. Its six concerns are incorporated below: evidence-weighted forecast framing, attribution-only production exposure floor, bidirectional look-count parity, Python 3.14 controller guards, paired backend parity, and explicit even-shape/stride preconditions.

## Milestones

### Milestone 1: Isolated Production Diff and Static Integrity
- [x] Confirm state is planning EXP-021, baseline query returns 94.15 at `7c1e7d8`, the integration/base commit is `7c1e7d8`, and the experiment branch is `autoresearch/maximize-cifar10-best-test-accuracy-021`.
- [x] In `BasicBlock.forward`, replace `shortcut[:, :, :: self.stride, :: self.stride]` only with explicit `F.avg_pool2d(shortcut, kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=False)`; retain the existing `F.pad` and every other production line.
- [x] Require the tracked diff to contain only `train.py`, exactly one production shortcut-expression replacement, no new module/parameter/buffer/configuration, and no change to `prepare.py`, dependencies, evaluator, seed, timer, logging, data policy, optimizer, or schedule.
- [x] Run syntax, Ruff, formatting/diff, parameter-count, state-dict, optimizer-group, and RNG invariants.

Verification commands:

```bash
bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv
git branch --show-current
git diff -- train.py
git diff --check
uv run ruff check train.py
uv run pre-commit run --files train.py
git status --short
```

### Milestone 2: Exact Shortcut Semantics
- [x] Create an ignored disposable controller at `experiments/021/preflight_pool_option_a.py`; it may read/import production code but must not alter tracked scope or call `Eval.evaluate()`. Resolve the repository root from `__file__`, prepend it to `sys.path`, keep process entry under `if __name__ == "__main__":`, and avoid starting forkserver workers during module import.
- [x] Prove runtime pool count is exactly two and restricted to `layer2[0]` and `layer3[0]`; all seven same-shape shortcuts remain direct identities and both residual stride-2 branches remain unchanged. Assert the complete `need_pad` inventory has exactly two blocks, both `stride==2`, and both receive even 32x32 or 16x16 spatial dimensions.
- [x] Verify transition tensors `32x32x32 -> 32x16x16 -> pad to 64x16x16` and `64x16x16 -> 64x8x8 -> pad to 128x8x8`.
- [x] Require candidate shortcut output to equal `F.pad(F.avg_pool2d(x, 2, 2), channel_pad)` exactly. Coordinate-ramp and four impulse tests must map the exact 2x2 cell to one mean; padded channels stay exactly zero.
- [x] Autograd must assign 0.25 to all four source pixels for the retained shortcut sum, while an accepted-control method assigns 1 only to the even phase. Hard- and probability-target forward/backward paths must be finite.
- [x] Construct aligned candidate/control models from reset seed 42, patching only the disposable control instances with the accepted shortcut method. Require 1,073,962 parameters, identical state keys/shapes/logical values, identical optimizer membership, and identical post-construction CPU/CUDA RNG.

Verification command:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/021/preflight_pool_option_a.py semantics --output .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/021/preflight-semantics.json
```

The controller exits nonzero on any mismatch, after serializing all diagnostics.

### Milestone 3: Replayable Production-Batch Safety
- [x] Materialize once, hash, and persist 200 exact post-N1/M7/CutMix batch-128 CPU tensors and targets from the production forkserver loader, including at least 80 hard and 80 probability-target batches. Serialize the corpus before assertions, explicitly shut all eight workers, and verify no worker survives.
- [x] In fresh deterministic processes, start control and candidate from bitwise-aligned seed-42 model/optimizer state and replay the same immutable corpus without regenerating transforms or CutMix.
- [x] Serialize per-step loss, debiased loss EMA, prediction histogram, maximum gradient/update norms, BN buffers, momentum-state finiteness, corpus hashes, target format, and terminal state before applying vetoes.
- [ ] Require all losses/gradients/parameters/BN buffers/momentum tensors finite, no candidate-only prediction histogram above 95% in one class, candidate terminal loss EMA `<=1.5x` control, hard/probability CE coverage, identical corpus hashes, and no RNG use by the candidate shortcut. **FAILED**: candidate concentration was 123/128 at step 17 and 128/128 at step 18 versus control 113/128 and 112/128; production vetoed.

Verification commands:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/021/preflight_pool_option_a.py materialize --batches 200 --output .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/021/preflight-corpus.pt --manifest .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/021/preflight-corpus.json
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/021/preflight_pool_option_a.py safety --corpus .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/021/preflight-corpus.pt --output .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/021/preflight-safety.json
```

This is a collapse veto only. Favorable short loss or prediction balance cannot authorize or select the candidate if another gate fails.

### Milestone 4: Fresh Paired Timing, Exposure, and Wall Projection
**Skipped after Milestone 3 research veto.**
- [ ] Confirm one idle H20 with 97,871 MiB and no competing process. Run one unscored device-conditioning subprocess, then five alternating fresh-process control/candidate pairs from identical logical weights and persisted production-distribution hard/soft batches.
- [ ] Each arm performs 100 warmups and 500 measured synchronized batch-128 steps. Measure full counted work plus forward, loss, backward, optimizer, p95, peak allocation, and inference separately; serialize every trial before aggregate assertions.
- [ ] Require candidate/control median trial-mean full-step ratio `<=1.02`, all finite, CV of trial means `<3%`, candidate p95 `<=1.05x`, projected steps `floor(26,898 * control_mean / candidate_mean) >=26,360`, peak allocation `<625 MiB` and no more than 16 MiB above control.
- [ ] Record and assert identical control/candidate values for deterministic-algorithm mode, cuDNN benchmark/deterministic/allow-TF32 flags, matmul TF32, dtype, device, and environment. `CUBLAS_WORKSPACE_CONFIG` may make diagnostics conservative versus production, but must be identical across both timing arms.
- [ ] Require candidate/control inference ratio `<=1.02`, projected end-to-end wall `<540s`, and projected evaluation count exactly 19. A projection of either fewer or more looks is a no-launch parity veto; no evaluator cadence edit is allowed in this representation experiment.

Verification commands:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/021/preflight_pool_option_a.py condition-device
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/021/preflight_pool_option_a.py timing --corpus .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/021/preflight-corpus.pt --pairs 5 --warmup 100 --steps 500 --output .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/021/preflight-timing.json
```

Any semantic, safety, timing, exposure, memory, inference, wall, or evaluation-count miss vetoes production. Do not change pool size, filter only one transition, add projection/BN, alter precision, enable compilation, or combine another idea as a rescue.

### Milestone 5: Single Fixed-Budget Production Run
**Skipped after Milestone 3 research veto. No `run.log` was launched.**
- [ ] Reconfirm the exact reviewed `train.py` diff, clean tracked scope apart from it, seed 42, one idle H20, no stale `run.log` variant, no surviving diagnostic worker/process, and all preflight JSON pass fields.
- [ ] Launch exactly once with required redirection under a 600-second hard supervisor. Do not stream full output, use `tee`, reroll, or retry a valid run.
- [ ] Monitor process liveness and only concise `run.log` tails/summary fields. Do not stop based on intermediate test accuracy; that would condition execution on the ground-truth metric.
- [ ] Record command, PID/exit code, start/end time, reviewed diff hash, hardware, and preflight artifact hashes in `03-execute.md`.

Production command:

```bash
rm -f run.log
timeout --signal=TERM --kill-after=15s 600s bash -c 'uv run train.py > run.log 2>&1'
```

### Milestone 6: Integrity and Metric Verification
**Skipped because production was not authorized.**
- [ ] Require exit zero, ten finite unique summary fields, `299.9<=training_seconds<=301.0`, `total_seconds <600`, `num_params=1,073,962`, one 80% switch, eight stopped workers, 45-55% strong CutMix, hard weak targets, at most one evaluation per epoch, no duplicate evaluation epoch, and at most 19 evaluations. Compare the actual count with EXP-010's 19; fewer looks are a deflation caveat on a miss, while more than 19 is a protocol-integrity failure.
- [ ] Record `num_steps` against the 26,360 mechanism-support floor. Falling below it weakens the registered exposure-preservation claim but does not invalidate a genuine `best_test_acc>=94.25%`; higher accuracy from fewer updates remains a formal improvement.
- [ ] Extract `best_test_acc`; formal improvement requires `>=94.25%`. A correct result below 94.25 is a valid no-improvement and must not be rerun.
- [ ] Compare switch/first-weak/final NLL/best with both EXP-010 and the pre-registered EXP-017 signature. Record mechanism interpretation separately from the formal verdict.
- [ ] Preserve `run.log` until the analyze phase has written `04-analysis.md` and indexed the result; then remove it before the next experiment.

## Code Changes

- **`train.py`**: In `BasicBlock.forward`, replace only transition shortcut spatial slicing with explicit nonoverlapping `F.avg_pool2d(..., kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=False)`. Retain the following zero-channel `F.pad`. This tests deterministic all-position transition aggregation while preserving Option-A channel semantics.
- **Ignored experiment artifacts under `.autoresearch/.../experiments/021/`**: Add the disposable semantic/safety/timing controller and its JSON/tensor evidence. These are protocol records, not production code, are never committed, and must never call the test evaluator.

Risks: average filtering can erase small edges or CutMix boundaries; direct shortcut gradients become 0.25 across four positions; residual and shortcut frequency content can mismatch; pooling kernels can reduce exposure. These risks are measured or diagnosed but not repaired within EXP-021.

## Configuration Changes

- Transition shortcut spatial reduction: `shortcut[:, :, ::2, ::2]` -> fixed `2x2`, stride-2 average pooling.
- Trainable parameters: unchanged at 1,073,962.
- All hyperparameters, FP32 dtype, seed, model depth/width, optimizer, decay, LR schedule, batch size, N1/M7, CutMix alpha/probability, 80% weak-tail transition, loader lifecycle, timer, and evaluator: unchanged.

## Execution Environment

- Method: local single-process production launch from the repository root; diagnostics use fresh local subprocesses and ignored artifacts.
- Resources: exactly one idle NVIDIA H20 with 97,871 MiB; batch 128; expected peak below 625 MiB; no concurrent GPU workload.
- Estimated runtime: semantic/safety diagnostics about 1-2 minutes, paired timing about 1-2 minutes, production about 330-340 seconds total; every individual production attempt hard-limited to 600 seconds.
- Log output: production stdout/stderr only in `run.log`; diagnostics only in named EXP-021 JSON/tensor artifacts; inspect concise tails or parsed fields only.
- Tool skill: none; this is a local run, not a remote job submission.

## Abort Criteria

- Stop before production on any failed semantic, state/RNG, exact-corpus, numerical, class-concentration, timing, exposure, memory, inference, projected-wall, projected-evaluation, scope, worker-lifecycle, or H20-idle gate.
- Terminate production at 600 seconds, on process exit nonzero, CUDA/driver/OOM error, non-finite training, missing/stalled process, or an unambiguous protocol violation. Treat timeout/crash as failure; fix and retry only if the cause is an implementation/infrastructure defect and the exact reviewed intervention remains unchanged.
- Never abort, rerun, or tune based on intermediate/final accuracy. Never use a different pool, one-transition variant, learned projection, precision change, seed, or fallback candidate in EXP-021.

## Verification Protocol

### Verification Procedure

1. Query and pin the baseline:

   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv
   ```

   Pass only for `baseline=94.15` and `baseline_commit=7c1e7d8`; the acceptance threshold is 94.25.

2. Confirm hardware immediately before diagnostics and production:

   ```bash
   nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
   ```

   Pass only for exactly one NVIDIA H20 near 97,871 MiB and no competing utilization/process. Resolve contention rather than measuring through it.

3. Require all Milestones 1-4 commands to exit zero and their serialized evidence to report every registered gate as passed. Diagnostics time out after 10 minutes each; production remains unauthorized after any miss.

4. Confirm scope and launch once:

   ```bash
   git diff --check
   git status --short
   rm -f run.log
   timeout --signal=TERM --kill-after=15s 600s bash -c 'uv run train.py > run.log 2>&1'
   ```

   Before launch, tracked changes must be only the reviewed `train.py` diff; `data/` is preserved. Exit 124/137, any other nonzero status, or wall time above 600 is failure.

5. Extract the summary without streaming the log:

   ```bash
   grep -E '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' run.log
   grep '^  eval ep' run.log
   grep '^augmentation_switch:' run.log
   ```

   If the summary grep is empty, inspect only `tail -n 50 run.log` for the crash. Require one finite value for each of ten fields, `best_test_acc>=94.25%`, `299.9<=training_seconds<=301.0`, total below 600, parameter/lifecycle gates above, unique evaluation epochs, and no more than 19 evaluation lines. Record actual steps and look count against 26,360 and 19 as attribution/deflation caveats; neither a sub-26,360 step count nor fewer than 19 looks invalidates an otherwise integral run. A result below 94.25 fails only the improvement condition and is recorded as no-improvement, not retried.

6. The analyze phase independently parses `run.log`, writes `04-analysis.md`, appends EXP-021 through `exp-index.sh`, and renders the verdict. Only after those durable records exist may `run.log` and its completed variants be removed.

### Informational Metrics (Optional)

- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params`: the unique final summary fields matched by the grep command above.
- switch accuracy, first weak accuracy, final/best gap, and evaluation count: parsed from `^  eval ep` and `^augmentation_switch:` lines in `run.log`; compare with EXP-010 and EXP-017 only after integrity passes.
- preflight full-step/inference ratios, projected exposure, peak allocation, corpus hash, loss EMA, prediction histograms, and worker shutdown: EXP-021 JSON manifests produced before production.
