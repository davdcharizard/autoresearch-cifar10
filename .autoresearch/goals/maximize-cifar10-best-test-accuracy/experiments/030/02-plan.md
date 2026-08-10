# Plan EXP-030: Raise the Weak-Tail Start LR to 0.02
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement and verify the isolated scalar
- [x] Create the experiment branch from the clean integration branch and change only `ANNEAL_START_LR = 0.01` to `0.02` in `train.py`.
- [x] Pass compile, Ruff, format, pre-commit, whitespace, scope, exact-diff, and static consumer checks.
- [x] Run a no-test schedule/boundary audit and a matched copied-state SGD recurrence sanity check; require unchanged strong LR, a monotone candidate tail, common endpoint, identical initial state/RNG, and approximately twofold first-weak parameter displacement to FP32 tolerance.

### Milestone 2: Execute one fixed-protocol run
- [x] Confirm exactly one idle NVIDIA H20 with approximately 97,871 MiB and remove no user data; require no stale completed `run.log` variant.
- [x] Run the candidate exactly once at seed 42 under the 600-second kill boundary, with all output redirected to `run.log`.
- [x] Monitor only bounded health/progress excerpts; do not rerun or tune the LR after observing trajectory data.

### Milestone 3: Verify metric and mechanism
- [x] Require exit zero, a complete finite ten-field summary, 300-second counted budget, summary total below the conservative 595-second process bound, one 80% switch with eight stopped workers, hard weak targets, unique evaluator epochs, and 1,073,962 parameters.
- [x] Require `best_test_acc >=94.25%` for improvement; preserve any valid lower result as no-improvement without retry.
- [x] Compare strong/switch, first-weak, full tail, final NLL, best-final gap, and exposure against EXP010 to distinguish useful extra tail motion from overshoot.

## Code Changes
- **`train.py`**: Change the single existing scalar `ANNEAL_START_LR` from `0.01` to `0.02`. The strong branch does not read this constant, so the candidate's program, initialization, complete LR-0.1 strong phase, N1/M7 and probability-0.5 alpha-1 CutMix, phase boundary, loader lifecycle, evaluation policy, model, seed, timer, and summary remain unchanged. This is program equivalence, not a claim that a new CUDA run reproduces EXP010's strong trajectory bitwise: the accepted protocol does not enable deterministic CUDA algorithms.

The unchanged tail expression is:

```text
q = (progress - 0.8) / 0.2
candidate_lr = 1e-4 + 0.5 * (0.02 - 1e-4) * (1 + cos(pi*q))
```

The first candidate weak update is approximately twice the accepted parameter displacement from an identical momentum/data state. Both schedules end at `1e-4`. This changes both loss-gradient motion and effective coupled-decay displacement; the experiment estimates their net accuracy effect and does not attribute a win to either alone.

## Configuration Changes
- `ANNEAL_START_LR`: `0.01 -> 0.02` (nearly double integrated LR during only the final 20% weak hard-label cosine tail).
- Expected candidate LR: `0.02` immediately above 80%, approximately `0.01005` at 90%, approximately `0.00301` at 95%, and `0.0001` at completion.
- Unchanged: `LR=0.1`, `LR_HOLD_FRACTION=0.8`, `MIN_LR=1e-4`, width-2 ResNet-20, batch 128, standard momentum 0.9, all-parameter weight decay `1e-4`, N1/M7, CutMix p=0.5/alpha=1, weak-tail transforms/targets, seed 42, workers, timer, and evaluator.

## Adversarial Review Response
- Mandatory Claude plan review completed with exit code 0; no fallback reviewer was used.
- Accepted the reproducibility caveat: the benchmark requires one seed-42 run but does not enable deterministic CUDA algorithms. A 94.25-94.35 result is a formal protocol improvement and will be merged, while being reported as weak evidence that may not reproduce within a 0.10-point margin; it does not authorize a confirmation rerun.
- Accepted source/tolerance corrections: all numeric anchors below cite `experiments/010/04-analysis.md`; evaluation integrity requires the unchanged early checkpoints, unique epochs, terminal coverage, and a plausible 18-20 looks rather than brittle equality to 19. Exposure and checkpoint drift are diagnostics unless corroborated by a protocol/infrastructure fault.
- Clarified that the copied-state 2x displacement check is only an arithmetic/scope sanity check. The actual production path is secured by an exact one-line diff plus AST/source inspection; the check is not evidence that the new LR improves optimization.
- Clarified retry policy: one unchanged rerun is permitted only after a documented external infrastructure failure that prevents a valid summary. A valid completed result is never rerun, even if it misses narrowly.

## Execution Environment
- Method: local `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1` from the project root.
- Resources: exactly one idle NVIDIA H20 with approximately 97,871 MiB; existing environment and eight DataLoader workers; no package changes.
- Estimated runtime: approximately 330-345 seconds total, exactly the fixed roughly 300-second counted budget, and around EXP010's 26,898 optimizer steps because the instruction graph is unchanged.
- Log output: full stdout/stderr only to project-root `run.log`; inspect bounded process/GPU/error/checkpoint summaries, never `tee` or full-log streaming.
- Tool skill: `/research-execute`; no remote submission, W&B, or timing campaign.

## Abort Criteria
- Before launch, stop for any tracked diff beyond the registered one-line scalar, baseline other than 94.15 at `7c1e7d8`, wrong/busy GPU, stale completed experiment log, schedule/state/SGD-recurrence failure, evaluator call change, malformed target/lifecycle check, or failing code-quality check.
- During execution, terminate for traceback, CUDA/OOM/resource error, non-finite loss/state, target assertion, worker-lifecycle failure, no bounded progress for 120 seconds while the process is live, or wall time reaching the external timeout.
- Do not abort for a low intermediate accuracy, an apparent overshoot, worse NLL, or a switch checkpoint below expectation; the complete fixed-budget run is required to render the metric verdict.
- Exactly one valid scored fixed-seed completion is allowed. One unchanged retry is allowed only for a documented external infrastructure failure that prevents a valid final summary; it must retain code/configuration/seed and is not a metric reroll. Do not rescue with 0.015/0.025, changed decay, clipping, a smooth transition, extra evaluation, another seed, or rerun of a valid result.

## Verification Protocol

### Verification Procedure
1. **Baseline and clean source (10s):** run `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`; require `baseline=94.15` and `baseline_commit=7c1e7d8`, so the improvement threshold is `94.25`. Require `sha256sum train.py` to match `git show 7c1e7d8:train.py | sha256sum` before editing and `git status --short` to contain no tracked modification; ignore and preserve untracked `data/`.
2. **Scope and quality (60s):** after editing, run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `uv run ruff format --check train.py`, `uv run pre-commit run --files train.py`, and `git diff --check -- train.py`. Inspect `git diff -- train.py`; require exactly `ANNEAL_START_LR = 0.01 -> 0.02` and no tracked file except `train.py`. Static inspection must find one definition of the constant, referenced only by the existing strict `progress > LR_HOLD_FRACTION` cosine branch; evaluator call sites and phase predicates must be unchanged.
3. **No-test schedule/state preflight (120s):** use an ignored experiment-scoped controller without `Eval.evaluate()`. AST/source inspection must bind the audited expression to the actual `train.py` branch and prove that the exact one-line scalar is the only production difference; the controller may reproduce the pure formula only after this binding check. Audit progress `0, .2, .4, .6, .7, .8, nextafter(.8,1), .85, .9, .95, 1.0`; require both arms exactly `0.1` through `.8`, candidate approximately `0.02` only above `.8`, monotone tail, and exact common `1e-4` endpoint. Require identical pre-CUDA initial parameter/buffer/RNG hashes, optimizer groups/state and 1,073,962 parameters. From copied model/momentum state and one fixed weak hard batch, require matching finite gradients/momentum buffers and aggregate plus representative candidate/control parameter-displacement ratio approximately `2.0` within FP32 tolerance. Record this as arithmetic/scope sanity only, not evidence for accuracy or stability beyond the first update.
4. **Environment/log gate (30s):** run `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader`; require exactly one NVIDIA H20, approximately 97,871 MiB total, no competing compute process, and idle utilization/memory consistent with the established machine. Require no stale `run.log` or renamed completed run log; remove only known experiment artifacts, never `data/`.
5. **Single production run (595s hard bound):** execute once with `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`; require exit status zero. Monitor with bounded `tail`, process status, `nvidia-smi`, and fatal-pattern searches. Timeout status 124/137 is failure; any other nonzero status is crash. A documented external infrastructure failure with no valid summary permits at most one unchanged retry; no retry follows a valid completion.
6. **Summary and protocol integrity (30s):** require exactly one finite numeric occurrence of each `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params`. Require `300.0 <= training_seconds < 301.0`, summary `total_seconds <595`, `num_params=1073962`, one `randaugment+cutmix->base` switch near 80%, eight stopped workers, 45-55% realized strong CutMix, hard weak targets, and no traceback/non-finite text. EXP010's sourced reference is 26,898 steps and 1,073,962 parameters (`experiments/010/04-analysis.md`, Results/Verification); 26,629 steps is the preregistered 99% exposure diagnostic. A lower count triggers investigation and is invalid only when corroborated by contention, lifecycle, timing, or scope failure rather than normal fixed-time jitter.
7. **Evaluation integrity and primary verdict (20s):** parse all `^  eval ep` records; require epoch IDs unique, no more than one evaluation per epoch, all unchanged 20/40/60/70% checkpoints represented, a terminal evaluation whose epoch equals `num_epochs`, and a plausible 18-20 total looks. EXP010 recorded 19 unique evaluations (`experiments/010/04-analysis.md`, Verification). Report any count asymmetry because `best_test_acc` is a maximum, but do not invalidate an otherwise compliant run solely for 18 or 20 looks. Parse `grep -E '^(best_test_acc|final_test_acc|final_test_loss):' run.log`; all integrity conditions plus `best_test_acc >=94.25` is improvement. Any valid lower score is no-improvement. A bare 94.25-94.35 pass is protocol-valid but explicitly weak reproducibility evidence under nondeterministic CUDA and cannot trigger a rerun.
8. **Mechanism diagnostics (20s):** compare the pre-switch/switch trajectory with EXP010's 89.73% final-strong checkpoint, the first weak checkpoint with 93.16%, final NLL with 0.1934, 26,898 steps, and 19 looks (`experiments/010/04-analysis.md`, Results/Verification), plus each common tail epoch and best/final gap. Since the accepted protocol does not force deterministic CUDA, a strong-phase difference is expected sampling evidence rather than automatic invalidation; only a large deviation (preregistered diagnostic: more than 1.0 point) plus corroborating scope/environment anomalies raises a validity concern. Faster first-weak recovery and a higher sustained tail support the mechanism; worse first-weak/loss or late regression supports overshoot. These diagnostics never override the primary threshold.

### Informational Metrics (Optional)
- Summary fields: `grep -E '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' run.log`.
- Phase/target/lifecycle provenance: bounded `rg 'switch|cutmix_batches|workers stopped|target' run.log` output.
- Evaluation trajectory: parsed `^  eval ep` lines in `run.log`, compared with EXP010's report/log evidence.
- Preflight schedule, state hashes, and update-ratio diagnostics: recorded in `experiments/030/03-execute.md` from the ignored controller output.
