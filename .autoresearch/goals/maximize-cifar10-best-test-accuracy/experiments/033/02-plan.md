# Plan EXP-033: Conservative Small-Area Random Erasing
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement and prove policy semantics
- [x] Create the experiment branch; modify only `train.py` with a top-level RNG-neutral erasing callable and exact strong-transform placement after ToTensor/before Normalize and CutMix.
- [x] Register p=0.25, scale 0.02-0.10, ratio 0.3-3.3, mean fill, non-inplace; keep weak/eval transforms and every accepted model/optimizer/schedule scalar unchanged.
- [x] Pass static quality, forced gate/mask/area/fill/label/RNG tests, target formats, parameter count, worker pickling, and exact scope.

### Milestone 2: Gate geometry, trajectory, and exposure
- [x] Materialize/hash exact pre-erasing sources plus registered erasing/CutMix decisions; require erased-example and effective-area distributions in preregistered bounds. Absolute production step count is informational; paired exposure retention is the pre-production gate.
- [ ] Replay 200 strong plus 64 weak exact batches in copied arms; require finite state, no candidate-only concentration, bounded loss/logit/gradient/update ratios, and exact provenance.
- [ ] Pass 5,000-batch eight-worker lifecycle/headroom and five-pair 1,000-step full-path timing gates; block production on >1% weighted overhead or any safety miss.

### Milestone 3: Execute and verify once
- [ ] Only after every gate passes, confirm one idle H20/no stale log and run seed 42 once under 595 seconds with redirected output.
- [ ] Require complete fixed-budget/lifecycle/policy/evaluator integrity and `best_test_acc>=94.25%`; never rerun a valid result.
- [ ] Diagnose switch fit, first weak, NLL, exposure, and realized erased/CutMix geometry versus EXP010.

## Code Changes
- **`train.py`**: add a top-level forkserver-picklable callable wrapping torchvision `RandomErasing(p=0.25, scale=(0.02,0.10), ratio=(0.3,3.3), value=mean, inplace=False)` inside `torch.random.fork_rng(devices=[])`, then insert it after `ToTensor()` and before `Normalize()` only in `strong_train_tf`.
- **`train.py`**: preserve the existing two-field collate/iterator contract and all target semantics. Do not add production provenance scans or metadata: exact erased masks and CutMix composition are proved in disposable preflight controllers, so the timed path measures only the intervention. Weak batches retain hard two-field behavior and no erasing.

## Configuration Changes
- Strong Random Erasing: absent -> p 0.25; requested area 2-10%; ratio 0.3-3.3; pre-normalization CIFAR-mean fill; non-inplace.
- Expected: 23.5-26.5% erased examples, conditional mean area 4.5-7.5%, unconditional area 1.1-1.9%; achieved per-erasure area 1.0-11.0% after integer rounding and final per-image effective erased area <=22% after registered CutMix overlap.
- Unchanged: width-2 ResNet-20, 1,073,962 params, N1/M7, CutMix p0.5/alpha1, 80% boundary, LR/momentum/decay, weak tail, seed, timer, evaluator.

## Execution Environment
- Method: ignored semantic/corpus/replay/worker/timing controllers; conditional local `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`.
- Resources: one idle 97,871-MiB H20; eight imported workers; existing environment/data.
- Estimated runtime: preflights up to 45-60 minutes under the individual gate timeouts, usually less with fail-fast ordering; production ~335 seconds if authorized.
- Log output: controller reports under `experiments/033/`; production only `run.log`; no `tee` and no production-time erasing telemetry.
- Tool skill: `/research-execute`; no remote/W&B.

## Abort Criteria
- Stop before production for policy/order/fill/area/RNG/target/provenance/corpus mismatch, nonfinite state, concentration/update/loss veto, worker starvation/lifecycle failure, or timing overhead >1%.
- During production stop for hardware/resource/fatal/nonfinite/lifecycle/target/policy assertion, no progress for 120s, or timeout. Do not abort for low intermediate accuracy/switch fit.
- One valid completion only; infrastructure-only retry at most once. No p/area/ratio/fill/placement/CutMix/worker/evaluator/seed rescue.

## Verification Protocol

### Verification Procedure
1. **Baseline/source (10s):** query the moving baseline with `exp-index.sh baseline`, record its value/commit (currently expected 94.15 at `7c1e7d8`), derive the improvement threshold as baseline+0.10, require pristine baseline `train.py`, imported `TIME_BUDGET_S=300`, imported worker count, and preserve untracked `data/`.
2. **Static/semantic (180s):** compile/Ruff/format/pre-commit/diff/AST. Force pass/fail masks; require exact outside/inside pixels, mean-fill -> normalized zero, unchanged labels and CPU/CUDA RNG, reproducible saved-state mask, correct sampled requested ranges, unchanged downstream accepted draws, weak policy, evaluator calls, params.
3. **Exact corpus (600s):** persist/hash 200 post-N1/M7/ToTensor pre-erasing source batches plus 64 weak batches and all gate/mask/CutMix geometry. Require 23-27% erased examples, >=99.5% placement success, achieved conditional area 1.0-11.0%, conditional mean 4.5-7.5%, unconditional 1.1-1.9%, final effective <=22%, 100 hard/100 CutMix decisions, valid targets.
4. **Trajectory safety (600s):** replay copied arms over all 264 records. Require finite state/positive BN variance/exact counters and provenance; no candidate-only >95% class share; per-step candidate/control loss/logit-RMS/gradient/update <=1.5; strong/weak terminal loss EMA <=1.25; serialize before assertions.
5. **Worker/lifecycle (300s):** 5,000 live strong deliveries with all imported workers; require 23.5-26.5% erased, geometry bounds, 48-52% CutMix, >=80% paired accepted throughput and >=140 batches/s, wait median<=0.5ms/p95<=1.5ms, all workers cleanly stopped, weak rebuild<5s and no weak erasing.
6. **Paired timing (1,500s):** five counterbalanced fresh-process pairs, 100 warmups and >=1,000 steps/arm, 40/40/20 strong-hard/strong-soft/weak weighting including loader/H2D/full GPU work. Require weighted mean ratio<=1.01, every pair<=1.03, CV<=2%, candidate p95<=1.04x control, paired projected exposure retention>=99%, peak<650MiB, wall projection<540s. Use paired ratio, not an absolute production-step floor, as the overhead gate. The per-image RNG fork is a load-bearing attribution control: retain it only if this gate passes; otherwise classify the candidate invalid rather than weakening the intervention after inspection.
7. **Production (595s):** confirm idle H20/no stale log/exact diff; run once. Require exit0, one finite summary, budget `[TIME_BUDGET_S,TIME_BUDGET_S+1)`, total<595, params, one 80% switch, expected workers stopped, 45-55% CutMix, hard/non-erased weak targets, no fatal signal. Policy geometry is established preflight rather than rescanned in production; steps are informational.
8. **Evaluator/verdict (20s):** require 18-19 unique looks including terminal, at most one evaluation per epoch, no duplicate epoch, and never more than the baseline's 19 looks. Integrity plus `best_test_acc >= moving_baseline+0.10` (currently 94.25%) is improvement; valid lower is no-improvement. Compare switch 89.73/87.08 marker, first weak93.16, NLL0.1934, steps26898, and preflight geometry; diagnostics cannot override metric. A bare threshold pass is protocol-valid but remains at the stated single-seed resolution limit and must not be over-interpreted causally.

### Informational Metrics (Optional)
- Corpus/live erase counts and requested/achieved/effective areas; CutMix overlap/targets; RNG hashes. Production has no erasing scan by design.
- Safety loss/logit/gradient/update/BN series; worker rates/waits; paired timing/CV/projection.
- Production summary/trajectory/provenance copied into `03-execute.md` before log removal.

## Adversarial Review Response
- Adopted the geometry arithmetic corrections: torchvision's rounded boxes are enclosed by 1.0-11.0%, and composed recipient/donor masks may reach 22%.
- Replaced the timing-dependent exact-19 evaluator gate with 18-19 unique terminal-inclusive looks and a strict ceiling of the baseline's 19 looks; this prevents extra max draws without rejecting a slightly slower valid run.
- Made paired exposure retention authoritative and production steps informational, resolving the proposal/plan conflict.
- Removed production mask scans and variable-length collate metadata. Exact provenance remains mandatory in preflight controllers, outside the measured intervention.
- Retained per-image `fork_rng(devices=[])` as a load-bearing attribution control because it preserves accepted augmentation draws; the timing gate decides feasibility before production.
- Corrected the preflight runtime estimate and made the baseline/threshold dynamically query the moving index.
