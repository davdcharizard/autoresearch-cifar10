# Plan EXP-034: Batch 512 With Fully Scaled LR
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement the exact large-batch operating point
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-034` from accepted commit `67c8e98`; modify only `train.py` in production and keep `prepare.py`/evaluator frozen.
- [x] Change exactly `BATCH_SIZE 256 -> 512`, `LR 0.2 -> 0.4`, `MIN_LR 0.002 -> 0.004`, and image-equivalent `MAX_STEPS 64000 -> 32000`; preserve every other source line and compile.

### Milestone 2: Prove construction, optimization, and worker semantics
- [x] Create ignored `experiments/034/preflight.py` with an independent `git show 67c8e98:train.py` oracle and guarded evaluator/test data; prove initial model/RNG identity, optimizer-group identity apart from doubled LR, 987,098 parameters, and exact four-line scope.
- [x] Prove doubled LR across the full time curve, image-equivalent cap, 97 batches/49,664 examples, finite FP32 batch-512 mixup and hard updates, safe H20 allocation, batch-shared coefficient semantics, accepted transform policy, and one-way exhausted-epoch RandAugment cutoff with exact clean-tail replay.

### Milestone 3: Require a material complete-body H20 gain
- [ ] Run balanced accepted-256/candidate-512 full-step timing for early mixup and hard regimes with raw output before assertions; require all CVs <=5%, image-rate ratio >=1.10, projected passes >=146.308096, and projected steps >=14,287.
- [ ] Persist the sole timing payload by exclusive atomic creation with commit/diff/device/session provenance for loader pacing; never overwrite it, rerun a stable throughput miss, lower the gate, or repair batch/LR/floor/momentum/warmup in this experiment.

### Milestone 4: Prove contemporaneous loader and wall feasibility
- [ ] Using the saved step medians, run fresh balanced accepted/candidate active/inactive real-loader epochs at arm-specific consumer pace; require correct batch/example counts, finite data, every CV <=5%, no worker failure/starvation, and correct one-way cutoff semantics.
- [ ] Compute active/hard phase epoch counts from their separate counted-time and step costs, then differential and conservative absolute wall projections with candidate evaluation count; require both <500 seconds and print all raw windows/projections before assertions.

### Milestone 5: Run and classify the sole fixed-seed score
- [ ] Reconfirm baseline 94.32 at `67c8e98`, one idle H20, local data, frozen evaluator, exact scope, no stale `run.log`, and all gates; execute exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [ ] Require exit 0, one finite summary, 300.0-300.1 counted seconds, total <600, 987,098 parameters, correct ordered transitions, unique once-per-epoch evaluations, and no numerical/CUDA/worker errors. Record realized passes and steps without rerunning if they miss projection.
- [ ] Classify improvement solely by `best_test_acc >=94.42%`; separately report final accuracy versus 94.22, final loss versus 0.2523, best-final gap, and evaluation count. Describe a best-only win as fixed-protocol metric improvement, not boundary-quality evidence, unless endpoint metrics corroborate it. Close exact `512, 0.4 -> 0.004, 32000-cap` after any stable gate miss or valid score without adjacent repair.

## Code Changes
- **`train.py` / constants only**: set batch size to 512, peak LR to 0.4, floor LR to 0.004, and maximum steps to 32,000. The cap remains exactly 16,384,000 possible images and both LR endpoints remain exactly twice accepted throughout the unchanged warmup/cosine function.
- **`.autoresearch/.../experiments/034/preflight.py`**: ignored verification-only harness adapted from EXP029 for source/construction semantics, batch-512 worker cutoff replay, full production-body H20 timing, and contemporaneous real-loader/wall timing. It must never call the real evaluator, construct test data, create `run.log`, or rerun throughput internally during loader timing.
- **`.autoresearch/.../experiments/034/throughput.json`**: ignored generated payload created exactly once via exclusive temporary file plus atomic rename and consumed read-only by loader timing. It includes accepted commit, exact candidate-diff SHA-256, constants, GPU name/UUID, UTC timestamp, session nonce, raw window counts/values, medians, CVs, image rates, retention, passes, steps, peak allocation, and `pass_status=true`. Throughput fails closed if the final path already exists; loader timing rejects missing/malformed/nonpassing or provenance-mismatched payloads.

## Configuration Changes
- Batch size: 256 -> 512.
- Peak/floor LR: `0.2/0.002 -> 0.4/0.004`, exact linear batch scaling with the accepted 100:1 ratio and time-curve shape.
- Maximum steps: 64,000 -> 32,000, preserving the image-equivalent safety cap.
- Epoch packing: 195 batches / 49,920 images -> 97 batches / 49,664 images; 336 examples are dropped per permutation instead of 80.
- Model/data/regularization: unchanged `(2,2,3)`, 987,098 FP32 parameters, Nesterov momentum 0.9, matrix-only decay `5e-4`, alpha-0.2 batch-shared mixup through 65%, early worker-private N1/M5 RandAugment through the first exhausted epoch at/after 65%, seed 42, crop/flip/normalization, and evaluator.

## Execution Environment
- Method: offline local semantic, GPU-throughput, and loader-wall preflights, followed by one local score only on pass; no network, remote, installs, W&B, GitHub, `gh`, fetch, push, or PR action.
- Resources: one NVIDIA H20, local CIFAR-10, installed environment, eight persistent forkserver workers; accepted peak was 1,096.3 MiB of 97,871 MiB.
- Estimated runtime: preflights under 5 minutes total; score about 350-380 seconds wall with a 600-second hard timeout.
- Log output: score stdout/stderr only in root `run.log`, retained through analysis then removed; ignored preflight JSON persists with EXP034 artifacts.
- Tool skill: none.

## Abort Criteria
- Abort before GPU timing on any scope/frozen-file/syntax failure; wrong constants or image-equivalent cap; construction/model/RNG/optimizer-group mismatch beyond intended LR; wrong topology/parameter count/dtype; non-finite update; unsafe memory; loader batch/count/transform error; mixup scalar/target mismatch; worker marker/private-RNG/clean-tail/cutoff error; evaluator/test access; or altered budget/evaluation control flow.
- Abort before loader timing if `throughput.json` or its intended final path already exists before the sole measurement; any GPU window is non-finite; any CV >5%; image-rate ratio <1.10; projected passes <146.308096; projected steps <14,287; atomic persistence fails; or timing provenance is malformed. Emit raw values before assertions; never repeat a stable miss.
- Abort before scoring on any contemporaneous loader error/starvation, wrong counts/shapes, non-finite data, any CV >5%, cutoff failure, or either wall projection >=500. Do not rerun a stable miss or rescue another batch/LR point.
- During score stop/classify on timeout, nonzero exit, OOM/worker/resource error, non-finite loss, no output for 60 seconds, malformed/missing/duplicate summary, wrong topology, invalid/repeated transition, duplicate evaluation epoch, or total >=600. Never rerun a valid completion or react to interim accuracy.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline`; require 94.32 at `67c8e98`, so the primary threshold is 94.42.
2. Within 30 seconds query `nvidia-smi`, local CIFAR presence, branch/status, `git diff --check`, `git diff --unified=0 67c8e98 -- train.py`, `git diff --exit-code 67c8e98 -- prepare.py`, and `uv run python -m py_compile train.py .../experiments/034/preflight.py`. Require one idle H20, only tracked `train.py`, and exactly the four constant changes.
3. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/034/preflight.py --semantics`. Guard `prepare.Eval` before accepted/candidate import and raise on `CIFAR10(train=False)`. Require accepted/candidate transform/model construction order and post-construction CPU/CUDA RNG identity, byte-equal initial model state, 987,098 FP32 parameters, and optimizer parameter-group membership/settings identity with only intended LR values doubled.
4. Require `BATCH_SIZE=512`, `MAX_STEPS*BATCH_SIZE=16,384,000`, `len(loader)=97`, 49,664 yielded examples, accepted loader flags, and candidate LR exactly `2 * accepted` at progress `0,.025,.05,.5,.65,1`. On fixed data, require finite `[512,10]` logits, scalar batch-shared mixup coefficient with aligned paired targets, finite paired and hard losses/gradients/Nesterov updates, and safe peak memory.
5. Build separately reset candidate and crop/flip-only batch-512 forkserver loaders from identical sampler generators, worker base seeds, dataset order, and task assignment. Instrument top-level picklable preflight wrappers to trace worker id, sample index, crop offsets, flip bit, target, active flag, and private RandAugment marker/state. Fully consume the same preceding active epoch in both arms before disabling only the candidate after exhaustion; require per-sample sampler/worker/crop/flip/target identity and exact clean-tail image replay, no marker leak, and no re-enable. Do not compare post-iterator RNG with accepted batch 256. Source checks must prove unchanged 300-second budget, at-most-once evaluation, one mixup transition, and one exhausted RandAugment transition.
6. Run `timeout 240s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/034/preflight.py --throughput`. For accepted batch256 and candidate batch512, separately measure early mixup and hard production bodies including pinned H2D, LR writes, zero-grad, real Beta/permutation/interpolation when active, forward, paired loss, finite guard, backward, Nesterov step, and final synchronization. Use >=20 warmups and three alternating balanced windows of >=50 steps per arm/regime from reset model/optimizer/RNG fixtures.
7. Print every raw window before assertions. Require finite values and population CV <=.05 in every arm/regime. From median seconds compute `accepted_rate=.65*256/accepted_mixup_s+.35*256/accepted_hard_s`, candidate analogously, `retention=candidate_rate/accepted_rate`, `projected_passes=133.00736*retention`, and `projected_steps=projected_passes*50000/512`; require retention >=1.10, passes >=146.308096, steps >=14,287, and safe peak allocation. Only after every gate passes, exclusively create a nonce-named temporary payload, flush/fsync it, and atomically rename to previously absent `throughput.json`; include the declared commit/diff/constants/device/timestamp/nonce/window-count/pass provenance. A pass establishes feasibility only.
8. Run `timeout 240s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/034/preflight.py --loader-timing`. Read the saved payload without remeasuring GPU steps; recompute and require its accepted commit, exact current diff hash/constants, GPU name/UUID, nonempty timestamp/session nonce, exact expected window counts, finite values, and `pass_status=true`. At each arm's active/hard median consumer pace, compare accepted and candidate real loaders in balanced order with one warm and three measured complete epochs per fresh arm/phase. Require accepted 195/49,920 and candidate 97/49,664 finite batches/examples, correct shapes/targets, cutoff behavior, every arm/phase CV <=.05, and no worker error.
9. For each arm/phase compute `epoch_stall=max(0,loader_median-batches*consumer_s)`. Derive phase epoch counts directly from counted phase seconds: accepted/candidate active epochs `195/(batches*active_step_s)` and hard epochs `105/(batches*hard_step_s)`, then total candidate epochs and conservative `candidate_eval_count=floor(total_candidate_epochs/5)+1`. Define `stall_delta=candidate_active_epochs*candidate_active_stall+candidate_hard_epochs*candidate_hard_stall-accepted_active_epochs*accepted_active_stall-accepted_hard_epochs*accepted_hard_stall`. Require `differential=345.3+max(0,stall_delta)+max(0,44.2*(candidate_eval_count/27-1)) <500` and `absolute=1.1+300+44.2*candidate_eval_count/27+candidate_active_epochs*candidate_active_stall+candidate_hard_epochs*candidate_hard_stall <500`.
10. Reconfirm exact audit and one idle H20, remove stale `run.log`, execute exactly `timeout 600s uv run train.py > run.log 2>&1` once, record PID/start, and never launch a second valid score.
11. Parse with `rg`; require one finite summary, 300.0-300.1 counted seconds, total <600, 987,098 parameters, no traceback/OOM/non-finite/worker error, one mixup transition at the first >=195-second step, and one later exhausted-epoch RandAugment transition with source-faithful lag `1 <= randaugment_step-mixup_step <=97`. Require unique every-fifth-epoch evaluations plus final partial epoch and report exact projected/realized counts.
12. Record realized passes as `num_steps*512/50000` and realized update count. A completed result below 146.308096 passes remains a valid goal result and may not be rerun, but the utilization mechanism is operationally inconclusive. Classify success solely by `best_test_acc >=94.42%`; report final accuracy versus 94.22, loss versus 0.2523, best-final gap, evaluation opportunities, dropped examples, VRAM, and counted/wall time as corroboration only. If best clears 94.42 only through a transient maximum while the endpoint does not improve, restrict the conclusion to fixed-protocol best-metric improvement and do not claim improved boundary quality or general large-batch superiority.

### Informational Metrics (Optional)
- Final summary: `run.log` - best/final accuracy, final loss, counted/total/startup seconds, epochs, steps, passes, VRAM, and parameters.
- Transitions/cadence: `run.log` - mixup/RandAugment epoch/step/time, transition lag, unique evaluation epochs/count, and best-final gap.
- GPU preflight: `throughput.json` - raw mixup/hard windows, medians/CVs, image rates, retention, projected passes/steps, and peak allocation.
- Loader preflight: direct output - raw accepted/candidate active/inactive epoch windows, medians/CVs/stalls, projected epochs/evaluations, and differential/absolute wall.
- Mechanism: realized versus projected passes/steps and best/final/loss deltas from accepted 94.32/94.22/0.2523.
