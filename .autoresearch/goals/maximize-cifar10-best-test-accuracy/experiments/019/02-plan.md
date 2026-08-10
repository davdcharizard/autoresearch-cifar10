# Plan EXP-019: Balanced Mixup and CutMix Geometry
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement the fixed geometry-and-strength swap
- [x] Modify only tracked `train.py`; confirm the pre-edit branch is `autoresearch/maximize-cifar10-best-test-accuracy-dev` at frontier commit `7c1e7d8` and preserve untracked `data/`.
- [x] Add alpha-0.4 Mixup and a single categorical strong-batch policy: 25% CutMix, 25% Mixup, 50% hard. Preserve alpha-1 CutMix, N1/M7, the 80% switch, and every non-data hyperparameter.
- [x] Return an integer geometry provenance from strong collation; replace the old `targets.ndim == 2` CutMix-count heuristic completely with kind-based validation/counters and conditional strong-three-tuple/weak-two-tuple unpacking.
- [x] Verify syntax, Ruff, AST/source invariants, exact model/optimizer configuration, and a semantic diff containing no unrelated change.

### Milestone 2: Pass deterministic semantics and production-distribution safety gates
- [x] Use an importable temporary controller outside the repository to prove forkserver-safe pickling, exact categorical probabilities, single-process collator CPU-RNG neutrality, accepted hard/CutMix equivalence, and Mixup pixel/target agreement.
- [x] Observe 20,000 eight-worker collations and require 48.5-51.5% total mixed, 23.5-26.5% CutMix, 23.5-26.5% Mixup, valid provenance, finite inputs, and valid hard/probability targets.
- [ ] Run paired accepted/candidate training from identical seed-42 model and optimizer state on 200 distinct real N1/M7 source batches driven by the same gate RNG stream; **failed the registered candidate-only >95% concentration veto on attempt 1**.
- [x] Shut down all eight strong workers explicitly, rebuild the weak loader in under five seconds, and prove its first batch remains a two-item FP32/int64 hard-label batch.

### Milestone 3: Protect fixed-budget exposure and wall time
- [ ] Confirm exactly one idle NVIDIA H20 with approximately 97,871 MiB memory.
- [ ] Run one unscored device-conditioning process, then five alternating fresh-process accepted/candidate timing pairs with 100 warmups and at least 1,000 real synchronized production steps per trial.
- [ ] Require candidate/control median-of-trial-means synchronized step ratio no greater than 1.01, CV below 3%, p95 ratio no greater than 1.04, and projected exposure of at least 26,629 steps.
- [ ] Require warmed loader delivery at least 1.20x GPU consumption, median iterator wait below 10% and p95 below 20% of candidate GPU-step time, integrated wall/count ratio no greater than 1.07 and no more than 0.02 above control, peak allocation below 650 MiB, and projected total wall time below 540 seconds.

### Milestone 4: Run the sole production experiment and validate it
- [ ] Remove any stale completed `run.log`, reconfirm the idle H20 and clean tracked scope, then run seed 42 exactly once as `uv run train.py > run.log 2>&1` under a 595-second TERM/5-second KILL supervisor.
- [ ] Poll only bounded summaries/tails while the job runs; do not stream full output and do not retry a valid run or change alpha/probabilities after observing accuracy.
- [ ] Parse the ten-field final summary, trajectory, geometry counts, lifecycle, evaluation epochs, exposure, VRAM, and timers; compare `best_test_acc` with 94.15% using the fixed 94.25% acceptance threshold.
- [ ] Record all controller and production results in `03-execute.md`; retain the intervention only if all integrity gates pass and the metric reaches at least 94.25%.

## Code Changes
- **`train.py`**: Change the strong-phase collator from a 50/50 hard/CutMix Bernoulli to one categorical draw with intervals `[0,0.25)` CutMix, `[0.25,0.50)` Mixup, and `[0.50,1)` hard. Instantiate module-level torchvision v2 MixUp with alpha 0.4, return a fixed integer provenance code, validate/unpack the strong three-tuple in the loop, count all geometries, and print those counts at the existing switch. The weak loader, loss call, model, optimizer, LR schedule, timer, and evaluator remain unchanged.
- **Temporary preflight controllers outside the repository**: Read-only diagnostics may import `train.py` after adding the project root to `sys.path`; they must be deleted after their gate. They are not production code and may not alter tracked files, dependencies, the dataset, or evaluator.

## Configuration Changes
- `CUTMIX_PROBABILITY`: `0.50 -> 0.25` (half of accepted regional-mixing events are replaced, not removed from the total mixed interval).
- `MIXUP_PROBABILITY`: absent `-> 0.25` (keeps total strong soft-target probability exactly 0.50 in expectation).
- `MIXUP_ALPHA`: absent `-> 0.40` (external idea review found alpha 0.2 too endpoint-heavy; a fixed two-million-draw probe measured central `[0.3,0.7]` mass 22.53% and minor-class mass 0.1604 at alpha 0.4 versus 13.37% and 0.1011 at alpha 0.2).
- Strong geometry policy: `50% alpha-1 CutMix / 50% hard -> 25% alpha-1 CutMix / 25% alpha-0.4 Mixup / 50% hard`.
- Attribution: this is a fixed compound geometry-and-strength bet, not an equal-regularization or one-lever comparison. A miss cannot distinguish absent Mixup complementarity from a poor alpha/split or net regularization drift; no valid-run tuning is permitted.
- Underfit risk: alpha 0.4 deliberately delivers more central interpolation than proposal alpha 0.2 and can repeat EXP-011-style strong-phase underfit even though mixed-batch probability stays 0.5. This risk is accepted before the run; the 87.08% switch marker is diagnostic only.
- All other constants and semantics: unchanged, including seed 42, batch 128, width-2 postactivation ResNet-20, 1,073,962 parameters, all-parameter SGD decay 1e-4, momentum 0.9, the 80% `lr=0.1` hold, the `0.01` cosine tail, 300 counted seconds, and existing evaluation cadence.

## Execution Environment
- Method: local single-process production run from the repository root; preflight uses fresh local processes and temporary importable controllers because Python 3.14 forkserver cannot safely launch from stdin.
- Resources: exactly one idle NVIDIA H20 GPU with approximately 97,871 MiB; eight persistent strong DataLoader workers; existing local CIFAR-10 data; no new package or dependency.
- Estimated runtime: semantic/safety gates 2-4 minutes, timing gates 3-5 minutes, production approximately 330-350 seconds; each individual production run has a strict 10-minute limit.
- Log output: production output only in `run.log` via redirection; bounded `tail`/`rg` polling; results transcribed to `03-execute.md`; completed log removed before the next experiment.
- Tool skill: `/research-execute` for local implementation, monitored execution, and artifact creation.

## Abort Criteria
- Any tracked modification outside `train.py`, any change to `prepare.py`, evaluator behavior, model/optimizer/schedule/weak-tail semantics, seed, dependency set, or maximum one evaluation per epoch.
- More than one categorical gate draw, use of Python/NumPy/CUDA RNG in the collator, loss of single-process collator CPU-RNG neutrality, a provenance/target mismatch, or loss of bitwise accepted behavior for controlled hard and retained CutMix branches.
- Any surviving use of target dimensionality to count geometry; dimensionality may only validate that `HARD` has `[B]` int64 targets and `CUTMIX`/`MIXUP` have `[B,10]` floating probability targets.
- Any 20,000-collation proportion outside its pre-registered interval, any non-finite/invalid target, or any failure to stop all eight strong workers and rebuild a hard-label weak loader in under five seconds.
- In the paired 200-batch probe: any non-finite input/target/logit/loss/gradient/parameter/BN/momentum state; fewer than 35 occurrences of any geometry; unequal total mixed decisions; candidate loss EMA above 1.5x control; or candidate-only greater-than-95% one-class concentration.
- Any timing/exposure/memory/wall gate miss. Do not rescue by moving mixing to GPU, changing workers/prefetch, changing alpha or probabilities, relaxing accounting, or forcing a backend.
- Production crash, nonzero exit, malformed/non-finite summary, more than 600 total seconds, missing/duplicate switch, non-hard weak target, repeated evaluation epoch, or external resource contention. A low intermediate accuracy is diagnostic and is not an early-stop or retuning signal.

## Verification Protocol

### Verification Procedure
1. Confirm the baseline and scope before implementation:
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv
   git status --short --branch
   git diff -- train.py
   ```
   Require baseline `94.15`, commit `7c1e7d8`, the integration branch, no tracked changes, and only preserved untracked `data/`. Timeout: 10 seconds.
2. After implementation, run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, a targeted source/AST controller, and `git diff --check -- train.py`. Require exit zero and prove that the only semantics changed are the declared strong collator, constants, provenance handling, and counters; parameter count and all accepted constants remain exact. Timeout: 30 seconds.
3. Run the temporary deterministic semantics controller. For fixed source tensors and seeds chosen to enter all three intervals, require bitwise CPU RNG restoration around direct single-process collator calls; bitwise candidate/control hard outputs for `u>=0.5`; bitwise candidate/control retained CutMix outputs and targets for `u<0.25`; valid alpha-0.4 Mixup pixel/target interpolation and roll pairing for `0.25<=u<0.5`; finite cross-entropy/gradients/momentum for all target forms; and no optimizer/model/CUDA RNG mutation before the declared training step. Timeout: 90 seconds.
4. Run the eight-worker 20,000-collation controller and paired 200-real-batch controller. Forkserver workers are checked through observable aggregate proportions, provenance, target validity, finiteness, and lifecycle rather than inaccessible internal RNG states. Require all Milestone 2 gates, including exact mixed-decision equality, loss-EMA ratio, and prediction concentration. Persist compact reports in `03-execute.md`; no test evaluator is called. Timeout: 300 seconds.
5. Run the unscored device conditioner and five alternating accepted/candidate timing pairs. Require every Milestone 3 threshold and use conservative projected steps `floor(26898 * control_mean / candidate_mean)`. No accuracy evaluation is permitted in timing. Timeout: 420 seconds.
6. Before production, run:
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
   test ! -e run.log
   git status --short
   ```
   Require one idle H20, no compute process, no stale log, only `train.py` modified, and preserved `data/`. Timeout: 10 seconds.
7. Launch the sole run as `timeout -k 5s 595s uv run train.py > run.log 2>&1`. Poll bounded lines only. Exit 124/137, any traceback, or elapsed wall above the supervisor is failure; never rerun a valid completed result. Timeout: 600 seconds.
8. Parse with bounded `rg`, `tail`, and an analysis script. Require exit zero; ten unique finite summary fields; `training_seconds` approximately 300; `total_seconds < 600`; `num_params == 1073962`; `num_steps >= 26629`; peak VRAM below the preflight ceiling; exactly one 80% switch with eight workers stopped; strong proportions in 48.5-51.5% hard and 23.5-26.5% each mixed geometry; weak targets hard; and at most one evaluation per unique epoch.
9. Metric verdict: improvement only when all integrity gates pass and `best_test_acc >= 94.25` (94.15 baseline plus required 0.10 percentage points). A finite lower value is `no-improvement`; no reroll, alternate alpha, probability change, rescue, or fallback is allowed. A protocol defect is invalid and may only be corrected without changing the reviewed intervention.

### Informational Metrics (Optional)
- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, `num_params`: final ten-field `run.log` summary parsed once the run exits.
- Switch and first-weak accuracy, best/final epoch, final NLL, evaluation count/epochs: bounded `rg '^  eval ep|^augmentation_switch:' run.log` plus a parser.
- Hard/CutMix/Mixup counts and fractions: the single `augmentation_switch` provenance summary.
- Timing exposure, iterator-wait percentiles, paired CV, peak allocation, safety loss ratio, prediction concentration, RNG/target checks: compact temporary-controller reports copied into `03-execute.md`.
