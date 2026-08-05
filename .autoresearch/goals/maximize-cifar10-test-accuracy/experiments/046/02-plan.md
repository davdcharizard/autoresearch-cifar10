# Plan EXP-046: CIFAR-Mean Crop Fill
- **Created**: 2026-07-27

## Milestones

### Milestone 1: Implement the exact constant-fill treatment
- [x] Create EXP046 branch from accepted `a7c42dc`; modify only `train.py` and freeze `prepare.py`/evaluator.
- [x] Add exactly `fill=(125, 123, 114)` to the existing `RandomCrop(32,padding=4)` without changing padding mode, width, transform order, normalization, or RandAugment.

### Milestone 2: Prove pixel, RNG, state, and boundary semantics
- [x] Add ignored evaluator-free `preflight.py`; prove transform topology, exhaustive 81-offset/two-flip pixel oracles, normalized fill values, and nonzero treatment incidence.
- [x] Prove accepted/candidate crop/flip and active/inactive RandAugment decisions plus terminal RNG states are identical; prove model/optimizer construction and all training controls remain accepted.

### Milestone 3: Protect persistent-worker wall feasibility
- [ ] Run sequential one-pool-at-a-time real-loader active/inactive timing with complete epochs; require stable delivery, identical sampler/target/decision traces, and projected total wall below500s.
- [ ] Run finite complete-step H20 smoke timing; require unchanged shapes/state/RNG law, at least127 projected passes, and peak allocation below2,048MiB.

### Milestone 4: Sole score
- [ ] Reaudit baseline94.48/source/resource/log and run exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [ ] Require a valid300-second completion with correct transitions/cadence and1,003,482 parameters; protected success additionally requires at least127 realized passes and best accuracy at least94.58%.

## Code Changes

- **`train.py` / `make_train_transform()`**: change the existing crop constructor to `transforms.RandomCrop(32, padding=4, fill=(125, 123, 114))`. This retains torchvision's constant PIL `ImageOps.expand` path and the same crop geometry/RNG while changing only synthetic border RGB.
- **ignored `experiments/046/preflight.py`**: block evaluator/test construction; load accepted source through `git show`; implement independent pixel/RNG/state/worker/timing oracles; print measurements before assertions; never read or write score metrics.

## Configuration Changes

- RandomCrop constant fill: scalar raw-black `0` -> fixed RGB `(125,123,114)`, obtained prospectively as rounded `255*(0.4914,0.4822,0.4465)` and already used by accepted RandAugment.
- Normalized candidate border is approximately `(-0.001204,+0.000153,+0.000559)` instead of `(-0.4914,-0.4822,-0.4465)` under accepted mean/unit-std normalization.
- Crop size/padding/padding mode, flip, transform order, RandAugment fields/private RNG, normalization, model/state, batch-shared mixup, optimizer, LR, budget, seed, evaluator, and cadence remain exact.

## Execution Environment

- Method: offline local semantic/loader/GPU gates followed by one local score; no network, remote, GitHub, install, W&B, or test-set design access.
- Resources: one idle NVIDIA H20, local CIFAR training data, installed `uv`,8 persistent forkserver workers.
- Estimated runtime: semantics <=180s, loader/timing <=240s, score about345s and killed at600s.
- Log output: preflight stdout; sole score project-root `run.log`, retained through verification/analysis and removed only before the next experiment.
- Tool skill: `/research-execute` only.

## Abort Criteria

- Abort before loader timing on any source scope, transform topology, pixel-mask, normalized value, crop/flip/RandAugment decision, RNG, sampler/state, model/optimizer, or control failure. Repair only an independently demonstrated verifier/implementation defect without changing fill, mode, width, order, or phase behavior.
- Abort before score if any retained loader epoch is malformed/nonfinite, sampler/target/decision traces differ, a window CV exceeds5%, a candidate phase median exceeds1.05x accepted, any candidate epoch exceeds1.10x accepted median, projected wall is>=500s, H20 projection is<127 passes, peak is>=2,048MiB, or the host/GPU is contended.
- Abort score on timeout/nonzero/OOM/worker/nonfinite/no-output60s/malformed summary/wrong device/state/cadence/transition. A structurally valid run below127 passes is recorded once as hypothesis failure, never invalidated or rerun.

## Verification Protocol

### Verification Procedure

1. Query the index baseline and require94.48 at `a7c42dc`; persist `git rev-parse HEAD`, branch, SHA-256 of current and `git show a7c42dc:{train.py,prepare.py,pyproject.toml}`, exact `git diff --name-only a7c42dc`=`train.py`, frozen-file diff, `git diff --check`, ignored harness, local data, absent log, compilation, `CUDA_VISIBLE_DEVICES`, CUDA count/name, `nvidia-smi -L`, utilization/memory/process output. Require EXP046 branch and exactly one idle NVIDIA H20.
2. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/046/preflight.py semantics`. Resolve project root in `sys.path`, replace `prepare.Eval`, block CIFAR test construction, and load accepted code from exact git source.
3. Diff/AST/source audit must allow only `fill=(125,123,114)` on the accepted crop. Require exact production order crop -> flip -> EarlyRandAugment -> tensor -> normalize; crop `(32,32)`, padding4, constant mode, no pad-if-needed; exact accepted RandAugment/normalization fields; no other constant, source, model, or training change.
4. Use an asymmetric RGB PIL fixture with no black/mean pixels. Force the installed production transform operations through all81 crop offsets and both flip decisions, but construct expected padding/crop/flip bytes and the padding-derived mask independently with preallocated NumPy RGB arrays and index slicing only - never PIL/torchvision padding or crop in the oracle. Require source pixels byte-exact, differences confined to the correctly flipped mask, accepted mask `(0,0,0)`, candidate mask `(125,123,114)`, exact no-difference at center offset, and intended nonzero differences at all80 touching offsets. Independently require the stated FP32 normalized vectors.
5. From cloned torch states, replay production accepted/candidate crop+flip at least256 times. Require identical sampled `(i,j)` and flip decisions, terminal torch RNG, shapes, targets, and independent pixel oracle. Run a prospectively seeded100,000-pair coordinate probe; print first and require padding contact in `[0.9865,0.9888]` around80/81 and empirical mean synthetic share in `[0.132,0.136]` around13.41%.
6. From cloned main and worker-private states, run at least64 production active RandAugment calls. Instrument the installed operation/sign/magnitude draws without adding production diagnostics; require identical decisions/private post-state/restored main state. Candidate pixels may differ outside the original mask after geometric transforms. With the flag inactive, require exact no-advance and crop-only pixel oracles.
7. Instantiate accepted/candidate model and optimizer from cloned seed42 CPU/CUDA states. Require exactly1,003,482 parameters/52 parameter tensors/97 state entries with identical names/order/bytes, optimizer groups/options/state, pooled-head seed36036 bytes, post-construction RNG, LR samples, mixup coefficient/permutation law, strict65% controls, finite guard, sole backward/step, and every-fifth-plus-final evaluator condition.
8. Before timing, run an independently qualified semantic worker replay, not used for performance. For each arm create one real training loader in production order with eight persistent workers and a trace wrapper returning index/target/worker/crop/flip/active/RandAugment decision/private-state hash plus tensor. Restore common construction states, exhaust one complete active epoch, flip the shared flag only after iterator exhaustion, then consume one inactive epoch from the same live pool. Require accepted/candidate nonpixel trace equality, exact sampler/target order, no active sample or private RandAugment advance in the inactive epoch, matching main/worker terminal RNG, finite tensors, and exact manual per-arm pixels. Tear pools down before timing.
9. Run `timeout 240s uv run python .../experiments/046/preflight.py timing` with production-equivalent `(tensor,target)` payload only - no trace metadata, hashes, or worker instrumentation. Use occurrence order `AA,CA,CI,AI,AI,CI,CA,AA`, first letter accepted/candidate and second active/inactive, with nearby common construction seeds `46000,46000,46001,46001,46002,46002,46003,46003`. At each occurrence keep one pool alive, exhaust one warm epoch, then retain three no-delay service epochs and three separate fixed11ms-consumer overlap epochs; tear down before the next occurrence. This yields six retained windows per arm/phase/mode across two streams.
10. Print all production-loader windows, batch/example counts, tensor shapes/finiteness, medians, CVs, paired ratios, phase-weighted delivery, and wall projection before gates. Require exactly195 batches/49,920 examples, each CV<=5%, candidate service and overlap medians<=1.05x controls, every candidate epoch<=1.10x its accepted median, and `343.9 + max(0,Cweighted_overlap-Aweighted_overlap)*130 <500s`, with weighted values in seconds per epoch. Instrumentation results from step8 cannot enter timing gates.
11. On fixed pinned accepted/candidate batches, run20 disposable warm steps per arm/regime, then exactly two counterbalanced retained blocks `AE,CE,AH,CH,CH,AH,CE,AE` with each window50 complete steps. Before every window create/load only that arm on GPU from common parameter/buffer bytes plus identical empty optimizer state, restore common input/RNG, synchronize, and delete it before the next arm. Print all16 raw windows, arm CVs, four early ratios, four hard ratios, four combined retentions, median projection, and peak. For pair index `i`, require exact fixed-time `retention_i=(0.65/CE_i+0.35/CH_i)/(0.65/AE_i+0.35/AH_i)`; require arm CVs<=5%, ratio CVs<=1%, every retention>=127/130.304, median projection>=127, and candidate peak<2,048MiB. Diagnostics cannot select another fill.
12. Immediately before score, re-run and persist every step1 source/resource/process audit; remove stale log; launch exactly `timeout 600s uv run train.py > run.log 2>&1` once. Require exit0, `Device: cuda`, one finite summary,300.0-300.1 counted,<600 wall,1,003,482 params, one ordered mixup/RandAugment transition, unique no-more-than-once-per-epoch evaluation cadence, and no errors. Retain `run.log` until analysis is complete.
13. Compute passes=`num_steps*256/50000`. Structural completion/resource validity is independent of exposure. Protected goal success requires both passes>=127 and `best_test_acc>=94.58`; any valid lower-exposure result remains the sole recorded no-improvement and cannot be rerun. Final accuracy/loss and input diagnostics are descriptive only. A normal-exposure miss closes exact always-on mean fill and immediate color/quantization/mode/width/order/schedule/RandAugment-fill/seed rescues.

### Informational Metrics (Optional)

- `run.log`: best/final accuracy, final loss, counted/wall/startup seconds, VRAM, epochs, steps/passes, params, transitions/evaluation cadence.
- Preflight: normalized border vectors, contact/synthetic-area incidence, loader windows/CVs/ratios/wall projection, GPU projection/peak.
