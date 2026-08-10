# Plan EXP-014: Calibrated stage-3 width-5 expansion
- **Created**: 2026-08-06

## Adversarial Idea-Review Resolution

- Fix one architecture before execution: stage widths `64/128/320`. There is no width-288 fallback, runtime switch, sweep, or post-gate resize; a valid failed width-320 preflight is recorded as a pre-metric failed leaf.
- Treat capacity limitation as unverified. Record candidate terminal debiased train loss and a paired early-conditioning trace as diagnostics, but do not claim a train-fit mechanism without a contemporaneous full parent control.
- Report paired parent/candidate `||epsilon||/||w||` for production-faithful SAM in preflight without retuning `rho=0.05`.
- Keep the formal `95.71%` threshold separate from a final-16 EMA mean `>=95.69%` scientific plateau target; report realized evaluation count and `best_test_acc - final16_mean` selection premium.
- Architecture-dependent constructor draws change the candidate's initialization and later CPU data-worker RNG state even with seed 42. This is an unavoidable fixed-seed package confound, not a searched seed: require candidate self-determinism, run no alternate initialization/data stream, and describe any narrow formal pass below the tail target as non-causal package evidence.

## Milestones

### Milestone 1: Implement the one fixed width taper
- [x] Change only the final two block specs from `128->256, 256->256` to `128->320, 320->320`; change final BN and classifier input from 256 to 320. Keep six blocks, stride order, projections, block implementation, and initialization method unchanged.
- [x] Print truthful architecture metadata: `PreAct WRN-16-[4,4,5]`, `stage_widths=64,128,320`, and computed `num_params=3827290`. Keep the independently computed MAC count `461556864` in the external harness/artifacts rather than hardcoding it into production.
- [x] Add only non-selective diagnostics needed for analysis: evaluation `charged_s`/`progress` in the existing outside-timer eval block and terminal `final_train_loss_ema` by reusing the already computed `debiased` scalar after the loop. Add no operation between per-step `t0` and `dt` for diagnostics. These values cannot stop a finite metric run, select a checkpoint, or change training.
- [x] Preserve all parent data, loss, optimizer, LR, drop-path, CutMix, SAM, EMA, seed, timing, evaluation, and summary behavior.

### Milestone 2: Prove deterministic architecture and parent-contract correctness
- [x] Materialize parent commit `d68f73a` under `/tmp`, hash-check it, import parent/candidate without invoking `main`, and prove candidate self-determinism under seed reset. Parent/candidate tensor equality and post-construction RNG equality are not required because shape-dependent constructor draws change the fixed-seed package.
- [x] Reconcile exact block/stride/projection/Conv inventory, parameter keys/shapes/count `3827290`, and MAC count `461556864`; require shape differences only in blocks 5-6, final BN, and classifier.
- [x] Run CPU FP32 and a separate candidate-only physical-GPU-0 BF16/channels-last forward/backward/Nesterov smoke. Reset CUDA peak statistics before constructing the candidate model, optimizer, SAM snapshots, and complete EMA shadow/restore state; require `(256,10)` finite logits, finite loss/BN state, finite nonzero gradients for every trainable tensor, and total candidate-only peak allocation `<4096 MiB`.
- [x] Exercise deterministic CutMix; require exactly six drop-path draws for a training forward with `drop_scale>0` and zero draws with `drop_scale=0`; run one ordinary update, one production-faithful SAM perturb/replay/BN-suppression/restore/update, 30 cadence-31 EMA samples across both parities, and one full-state EMA evaluate/swap/restore. Require exact restoration, one BN/optimizer update, complete coverage, balanced cadence, no RNG/state failure, and report paired `||epsilon||/||w||` without changing rho.

### Milestone 3: Run the first complete accuracy-blind GPU-0 preflight
- [x] Confirm physical GPU 0 is the approximately `97871 MiB` NVIDIA H20 and expose exactly one visible H20 with `CUDA_VISIBLE_DEVICES=0` before every GPU command.
- [x] Immediately after importing parent and candidate, monkeypatch both module-level evaluator objects so any `evaluator.evaluate` call raises. Then run one paired 200-step real-CIFAR conditioning trace. Each arm is separately seed-42 deterministic; share each materialized transformed batch and scripted CutMix/drop-path decisions. Record finite parent/candidate losses and activation/gradient norms at fixed steps. Numerical equality is neither expected nor a selection gate.
- [x] Run five alternating-order warmed timing rounds per arm with exactly 100 early ordinary, 40 early CutMix, 20 late ordinary, 20 late SAM steps, and 40 eval-mode forwards on fixed synthetic tensors per round. Include actual optimizer, SAM, and cadence-31 EMA work; the CIFAR test loader may exist from import but is never iterated and test accuracy cannot be produced.
- [x] Weight charged paths by EXP-011 counts `10512/25798`, `10345/25798`, `2470/25798`, and `2471/25798`. On the first complete measurement require parent weighted-round drift `(max-min)/median <=0.03`, paired-ratio median absolute deviation divided by median `<=0.01`, median candidate/parent ratio `<=1.15`, and `max(round_ratios) <=1.20`. Derive and require as arithmetic consistency checks: steps `25798/median_ratio >=22000`, complete epochs `floor(projected_steps/195) >=112`, EMA samples `160*projected_steps/25798 >=130`, and conservative total `447.9*max(1,median_ratio) <600s`. Report joint-process allocation only as informational; the binding memory gate comes from the separate candidate-only smoke.
- [x] The complete gate result is decisive. Do not rerun timing, resize to 288, change batch/LR/decay/SAM/EMA, or inspect accuracy after a failure. A terminal preflight rejection produces no metric launch and is recorded as `crash`/`NaN` under the tree convention.
- [x] Harness correction is allowed only for a Python/shell exception, missing/wrong file, or demonstrably malformed assertion before any numeric gate vector is emitted. Once any round/gate number is emitted, every numeric failure (including parent drift/dispersion) is decisive and can never be relabeled as a harness error.

### Milestone 4: Execute exactly one fixed-seed metric run
- [x] Remove stale `run.log`, reconfirm physical GPU 0, and launch exactly once: `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`. **Skipped as required after the decisive preflight rejection; no metric launch occurred.**
- [x] Monitor process/GPU/log liveness without pruning on finite loss, train accuracy, intermediate test accuracy, or dose. Abort only on exception, CUDA/OOM, nonfinite/integrity failure, EMA/SAM restoration failure, 120 seconds without process/GPU/log progress, or outer timeout. **Metric monitoring was not reached after the mandatory preflight stop.**
- [x] Preserve `run.log` and all preflight/smoke evidence through durable transcription, Claude-only raw-result review, analysis, commit, and tree insertion. Never retry the metric or alter width/hyperparameters from the result. **No `run.log` existed; all completed preflight evidence was transcribed and Claude-reviewed.**

### Milestone 5: Verify and classify
- [x] Apply integrity-first precedence: hard-constraint/state/RNG/evaluation corruption is `invalid/NaN`; no result or nonzero process failure is `crash/NaN`; only exit 0 with intact evidence can be improvement/no-improvement.
- [x] Require exit 0, charged seconds `[299.5,301.0]`, total `<600s`, exactly one evaluation per epoch, complete summary, `num_params=3827290`, only tracked `train.py` changed, and no traceback/CUDA/OOM/audit/RuntimeError/NaN/Inf signature. **Metric conditions were skipped after the failed necessary preflight condition.**
- [x] Formal improvement is `best_test_acc >=95.71%` versus parent `95.61%`. Because the fixed evaluator has exactly 10,000 test examples, this is exactly at least 9,571 correct and the two-decimal output cannot round a sub-threshold count upward. A valid value below `95.71%` is one no-improvement and never authorizes another width, LR, SAM radius, or seed. **No accuracy was measured.**
- [x] Mechanism support additionally requires at least `22000` steps, at least 130 EMA samples with ordinary/SAM imbalance at most one, final-16 EMA mean `>=95.69%`, exact state/restoration audits, and intact CutMix/SAM/EMA dose. Report candidate terminal debiased train loss, evaluation count, tail range/final, and max-minus-tail premium. Scientific shortfall does not override the formal tree verdict. **Not evaluated because the metric run was not launched.**

## Code Changes

- **`train.py`**: Change the two final-stage channel shapes and tail BN/classifier dimension; update architecture/config/parameter reporting; add outside-charged evaluation progress and terminal debiased-train-loss diagnostics. No other tracked file changes. MAC accounting remains in `/tmp` verification code and the durable plan/execute artifacts.

The block class, module count/order, forward control flow, initialization function, and all training mechanisms remain unchanged. Shape-dependent initialization changes both weights and the later global CPU data-worker realization, so the candidate is evaluated as one deterministic seed-42 architecture package, not an isolated same-stream width effect. No overlapping parent weights are copied, RNG draws burned, alternate seed/data stream run, or post-result reroll attempted.

## Configuration Changes

- Stage widths: `64/128/256 -> 64/128/320` (adds semantic capacity only at `8x8`, preserving early/middle processing).
- Stored/trainable parameters: `2748890 -> 3827290` (`+39.23%`).
- Conv/Linear MACs per image: `392612352 -> 461556864` (`1.1756045x`).
- Every other constant remains exactly EXP-011, including `PEAK_LR=0.2`, `WEIGHT_DECAY=1e-4`, `MAX_DROP_PATH=0.08`, CutMix, `SAM_RHO=0.05`, and EMA half-life/cadence.

## Execution Environment

- Method: local deterministic CPU checks, local physical-GPU-0 integration and one-shot paired preflight, then at most one local fixed-seed metric run.
- Resources: `CUDA_VISIBLE_DEVICES=0`; one NVIDIA H20 with approximately `97871 MiB`; existing `uv` environment; no dependency installation.
- Estimated runtime: about 2-4 minutes for checks/preflight and roughly 470-540 seconds for the metric process; every GPU workload is bounded and the metric run has a 600-second outer timeout.
- Log output: transient `/tmp/exp014_*` files for parent snapshot/harness/preflight; repository-root `run.log` for the metric. Exact values are copied into `03-execute.md` before cleanup.
- Tool skill: local execution; no remote platform or W&B.

## Abort Criteria

- Wrong physical or visible GPU; tracked change outside `train.py`; dependency/evaluator/seed/budget/evaluation-cadence change; any width other than fixed `64/128/320`; any LR, weight-decay, CutMix, drop-path, SAM, or EMA retuning.
- Architecture inventory, parameter/MAC reconciliation, output shape, finite gradient, CutMix RNG, early six-draw/terminal zero-draw drop-path, SAM perturb/replay/BN/restore, EMA cadence/coverage/swap/restore, optimizer identity, or state assertion failure.
- Candidate-only GPU-smoke peak allocation `>=4096 MiB`; or first complete preflight parent drift `>0.03`, ratio dispersion `>0.01`, median ratio `>1.15`, maximum round ratio `>1.20`, inconsistent derived projection, or any nonfinite/collapse/integrity failure. Finite conditioning differences cannot abort or alter the recipe.
- Metric-run exception, CUDA/OOM, nonfinite/integrity signature, 120 seconds without process/GPU/log progress, or 600-second timeout. Intermediate accuracy, finite loss, realized dose, and train-fit diagnostics cannot trigger pruning.
- A completed preflight or metric measurement is never rerun. Before any numeric gate output, only syntax/runtime exception, missing/wrong file, or demonstrably malformed assertion qualifies for the execution skill's bounded harness/code retry; numeric drift, dispersion, latency, memory, projection, or model-state results never qualify.

## Verification Protocol

### Verification Procedure

1. Query parent with `bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 011`; require metric `95.61`, formal threshold `95.71`, reference final-16 mean `95.493125`, `25798` steps, 160 EMA samples, and 133 evaluations from EXP-011's report.
2. Before every GPU command run `nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader`, then under `CUDA_VISIBLE_DEVICES=0` require `torch.cuda.device_count()==1` and visible name `NVIDIA H20`.
3. Run `uv run python -m py_compile train.py`, `git diff --check`, `git diff --name-only d68f73a`, and `git status --short --untracked-files=all`; require exactly `train.py` as the tracked code change. All harnesses/snapshots live under `/tmp`; `run.log` is the sole repository-local ignored raw artifact during the metric run and is tracked separately by explicit existence/stat checks rather than `git status`.
4. Materialize/hash-check `git show d68f73a:train.py` in `/tmp`, run all Milestone-2 smokes under bounded timeouts, and execute the single complete Milestone-3 preflight under `timeout 420s` with `CUDA_VISIBLE_DEVICES=0`. Do not query test accuracy. An incomplete outer-timeout result is pre-metric `crash/NaN`, not permission to reduce the workload and rerun.
5. Only after every gate passes, remove stale `run.log` and launch once under the exact Milestone-4 command. Preserve the process exit status and raw evidence.
6. Require `rg -c '  eval ep' run.log == num_epochs`, live+EMA source counts equal epochs, and no unexplained match from `rg -n -i 'traceback|cuda error|out of memory|ema_audit_failed|runtimeerror|(^|[^a-z])(nan|inf)([^a-z]|$)' run.log`.
7. Extract the full summary, bare `params=3827290` value from the config line, comma-formatted summary count, architecture label, final train-loss diagnostic, CutMix/SAM/EMA audits, final 16 EMA evaluations with mean/range/progress, evaluation count, and max-minus-tail premium. Apply formal and scientific thresholds separately; charged time is structurally at least 300 seconds, while `<=301.0` detects excessive last-step overshoot.
8. Run Claude as the sole raw-result adversarial reviewer. Correct evidence wording, finish analysis/tree insertion, verify commit scope, then delete `run.log`, parent snapshot, and all temporary EXP-014 harnesses before the next loop.

### Informational Metrics (Optional)

- Summary: final accuracy/loss, terminal debiased train loss, training/total/startup seconds, peak VRAM, epochs/evaluations, steps, and parameters.
- Architecture/dose: stage widths, MACs, CutMix applied/eligible, SAM applied/eligible/start, EMA updates/parity/decays/distances/swaps/restores, and paired relative SAM perturbation.
- Stability: final-16 EMA values, mean/min/max/range/progress span, final accuracy, best epoch, and `best_test_acc - final16_mean`.
