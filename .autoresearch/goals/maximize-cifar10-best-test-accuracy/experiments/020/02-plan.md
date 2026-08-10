# Plan EXP-020: Isolated PyTorch Nesterov Momentum
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Apply and prove the one-keyword intervention
- [x] Create the EXP020 branch from integration commit `7c1e7d8`, preserve untracked `data/`, and modify only tracked `train.py`.
- [x] Add only `nesterov=True` to the accepted `optim.SGD(...)` constructor; preserve momentum 0.9, dampening zero, coupled all-parameter decay 1e-4, the single parameter group, and every non-optimizer semantic.
- [x] Pass syntax, Ruff, diff/AST, parameter-count, source-hash, optimizer-default, model/data/schedule/timer/evaluator/lifecycle, and RNG-construction checks.
- [x] Match installed PyTorch 2.9.1 ordinary/Nesterov recurrence to a manual FP32 controller across changing gradients, nonzero coupled decay, and an LR change; prove exact first-step buffer equality and required 1.9x Nesterov direction directly on the pre-storage update tensor, never from rounded parameter differences.

### Milestone 2: Pass replayable production-distribution safety
- [x] Materialize one exact corpus before either optimizer arm: 200 distinct post-N1/M7 plus accepted p=0.5 CutMix batches, explicitly stop all eight workers, rebuild the weak loader, and materialize 64 hard crop/flip batches.
- [x] Persist actual post-transform FP32 inputs/targets outside the repository, record SHA-256 and target/mix metadata before assertions, and reuse the same immutable digest for any mechanical controller retry.
- [x] Train bitwise-aligned ordinary/Nesterov models on the same corpus in one process: 200 strong steps at LR 0.1, then 64 weak steps at LR 0.01 without resetting momentum.
- [ ] Require all manual-state, finiteness, loss-EMA, update-spike, class-concentration, hard/soft target, transition, worker-lifecycle, parameter-identity, and RNG gates before timing. **Failed candidate-only >95% concentration at strong step 11.**

### Milestone 3: Protect fixed-budget exposure
- [ ] Confirm exactly one idle NVIDIA H20 with approximately 97,871 MiB and no compute process.
- [ ] Run one unscored device-conditioning subprocess, then five alternating fresh-process accepted/Nesterov timing pairs over the same persisted corpus, with 100 warmups and at least 1,000 synchronized complete steps per trial.
- [ ] Require candidate/control median-of-trial-means ratio at most 1.01, CV below 2%, candidate p95 at most 1.04x control, peak allocation below 610 MiB and no more than 8 MiB over control, and projected exposure at least 26,629 steps.
- [ ] Require finite state, unchanged data/evaluation lifecycle, and conservative total-wall projection below 540 seconds; no backend forcing or uncounted optimizer work.

### Milestone 4: Run once and verify
- [ ] Remove stale completed `run.log`, reconfirm idle H20/scope/intervention, and launch seed 42 exactly once as `uv run train.py > run.log 2>&1` under a 595-second TERM/5-second KILL supervisor.
- [ ] Poll only bounded summaries/tails; do not stream full output, reroll, retune, warm up, clip, reset momentum, or fall back to ordinary momentum after a valid result.
- [ ] Parse the complete summary and trajectory; require 300 counted seconds, total below 600, at least 26,629 steps, 1,073,962 parameters, one valid switch, hard weak targets, unique at-most-once-per-epoch evaluations, and no more than EXP-010's 19 total evaluation looks.
- [ ] Accept only `best_test_acc >=94.25%`; record switch/first-weak/best/final/NLL, exposure, memory, timing, and every integrity result in `03-execute.md`.

## Code Changes
- **`train.py`**: Add the literal keyword `nesterov=True` to the existing single `optim.SGD` call. No helper, diagnostic, logging, model, data, schedule, loss, timer, worker, checkpoint, evaluator, or summary code enters production.
- **Temporary controllers/artifact outside the repository**: Importable scripts prepend the project root explicitly, serialize aggregate evidence before assertions, and persist exact post-transform training tensors plus digest for replay. They are deleted after execution and never invoke the test evaluator.

## Configuration Changes
- PyTorch SGD `nesterov`: default `False -> True`.
- Momentum: unchanged `0.9`; dampening remains installed default `0`; maximize, foreach, fused, differentiable, and decoupled-decay behavior remain installed defaults.
- Coupled weight decay: unchanged `1e-4` on all parameters in the same group.
- All other configuration: unchanged, including seed 42, FP32, batch 128, width-2 postactivation ResNet-20, p=0.5 alpha-1 CutMix during N1/M7 through 80%, hard weak tail, LR 0.1 hold, 0.01-to-1e-4 cosine, 300 counted seconds, and fixed evaluator cadence.

## Execution Environment
- Method: local one-GPU controllers and one production process from repository root; temporary path-launched controllers prepend the root for Python 3.14 imports.
- Resources: exactly one idle NVIDIA H20, approximately 97,871 MiB; eight persistent loader workers only during corpus materialization/production; existing CIFAR-10 data; no dependency changes.
- Estimated runtime: recurrence/safety 1-3 minutes, timing 3-5 minutes, production approximately 330-350 seconds; hard production limit under 10 minutes.
- Log output: production only to `run.log` by redirection; bounded `rg`/`tail` monitoring; all ephemeral results transcribed with values into `03-execute.md`; completed log removed before the next experiment.
- Tool skill: `/research-execute` for local implementation, monitored run, and verification.

## Abort Criteria
- Any tracked semantic change beyond `nesterov=True`, any modification outside `train.py`, or any change to model/data/schedule/loss/timer/evaluator/seed/worker lifecycle/summary.
- Any optimizer setting other than momentum 0.9, dampening zero, Nesterov true, coupled decay 1e-4, and the installed defaults on one all-parameter group.
- Manual recurrence mismatch; unequal initial models/logits/losses/raw gradients/RNG; unequal first momentum buffers; first direction not 1.9x ordinary before parameter-storage rounding; changed parameter identities/state keys/shapes/dtypes/devices; or optimizer RNG consumption.
- Safety corpus not persisted before either arm, changed digest on retry, fewer than 200 distinct strong or 64 weak batches, strong mixed fraction outside 45-55%, incomplete eight-worker shutdown, non-hard weak targets, or threshold evidence not emitted before assertion.
- Across 264 paired steps: any non-finite state; first-batch Nesterov replay loss above 2x pre-update; candidate-only greater-than-95% one-class concentration; post-initial update spike above 10x preceding 16-step median; strong steps 101-200 loss EMA above 1.5x control; or first-eight-weak loss EMA above 2x control.
- Any timing/exposure/memory/wall threshold miss. Do not rescue by forcing foreach/fused, lowering LR, adding warmup/clipping, changing decay, resetting momentum, or altering the corpus.
- Production crash, timeout, malformed summary, scope drift, lifecycle fault, duplicate evaluation epoch, more than 19 total evaluations, or invalid timer. Low accuracy/switch fit is diagnostic and never authorizes early stop or rerun.

## Verification Protocol

### Verification Procedure
1. Confirm baseline/branch/scope with `exp-index.sh baseline`, `git status --short --branch`, `git rev-parse --short HEAD`, and `git diff -- train.py`. Require 94.15 at `7c1e7d8`, integration/experiment branch as appropriate, no tracked pre-edit changes, and preserved `data/`. Timeout: 10 seconds.
2. After implementation, run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `git diff --check -- train.py`, and an AST/source controller. Require exit zero, exactly one added keyword, unchanged file semantics elsewhere, one SGD group, exact accepted constants, and 1,073,962 parameters. Timeout: 30 seconds.
3. Run a direct optimizer semantics controller over at least three FP32 steps with changing gradients, coupled decay, and an LR change. Match installed ordinary/Nesterov buffers and parameters to manual recurrence; require bitwise aligned reset state, first buffer `grad + 1e-4*theta0`, Nesterov direction ratio 1.9 measured on the explicit pre-storage update tensor, finite later state, and no CPU/CUDA RNG consumption. Timeout: 60 seconds.
4. Materialize and hash one 200-strong/64-weak production-distribution corpus, then run paired safety from that immutable file. Planning diagnostic `MIXCOUNT_PASS` already confirmed seed-42 materialization at exactly 100/200 CutMix batches with all eight workers stopped, closing the no-reroll window precondition. Require 45-55% strong CutMix, hard weak targets, all registered manual/finite/concentration/update/loss/transition gates, and serialized evidence preceding any assertion. No test evaluation occurs. Timeout: 300 seconds.
5. Run device conditioning and five alternating fresh-process timing pairs on the persisted corpus. Require median ratio `<=1.01`, each CV `<=2%`, p95 ratio `<=1.04`, projected steps `floor(26898*control_mean/candidate_mean) >=26629`, peak `<610 MiB` and candidate delta `<=8 MiB`, finite state, and projected total `<540s`. Timeout: 420 seconds.
6. Before production, require one idle H20/no compute process via bounded `nvidia-smi`, absence of `run.log`, experiment branch at `7c1e7d8`, and a diff containing only `nesterov=True`. Timeout: 10 seconds.
7. Launch `timeout -k 5s 595s uv run train.py > run.log 2>&1`. Poll bounded error/evaluation/progress lines. Exit 124/137, traceback, resource contention, or supervisor expiry is failure; never rerun a valid completed result. Timeout: 600 seconds.
8. Parse the ten unique finite summary fields and bounded trajectory. Require exit zero; approximately 300 counted seconds; total `<600`; params `1073962`; steps `>=26629`; one switch near 80% with eight workers stopped; 45-55% CutMix; hard weak phase; unique at-most-once-per-epoch evaluations; and total evaluations `<=19`. A higher count is evaluation-confounded and cannot be accepted as improvement.
9. Metric verdict: improvement only if all gates pass and `best_test_acc >=94.25` (94.15 plus required 0.10 points). A finite lower result is no-improvement with no retry. A protocol defect is invalid and may be corrected only with the one-keyword intervention unchanged.

### Informational Metrics (Optional)
- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, `num_params`: final `run.log` summary.
- Switch, first weak, best/final epoch, NLL, best-final gap, evaluation epochs/count: bounded `rg '^  eval ep|^augmentation_switch:' run.log` plus parser.
- Recurrence error, first-direction ratio, persisted-corpus SHA/counts, paired loss/concentration/update diagnostics, timing ratios/CV/p95, projected exposure, memory: controller summaries copied inline to `03-execute.md`.
- Any bare 94.25-94.30 pass is recorded as fragile single-seed evidence near the ten-image resolution floor; preflight integrity does not remove final trajectory variance.
