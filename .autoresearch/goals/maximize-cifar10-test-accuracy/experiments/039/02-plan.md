# Plan EXP-039: Rephase Cosine Across the Hard-Label Tail
- **Created**: 2026-07-27

## Milestones

### Milestone 1: Implement the exact boundary-derived curve
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-039` from accepted `a7c42dc`; modify only `train.py` and freeze `prepare.py`/evaluator.
- [x] Change only `learning_rate()`: preserve the accepted curve through 65%, then cosine from the accepted 65% LR over the remaining 35% to accepted `MIN_LR=0.002`, with no new constant, momentum change, or runtime state.

### Milestone 2: Prove schedule and update semantics
- [x] Add ignored evaluator-free `experiments/039/preflight.py` using independent `git show a7c42dc:train.py`; prove exact source scope, unchanged model/optimizer/data/RNG state, 1,003,482 parameters, and independent formulas on dense and boundary grids. The first harness attempt correctly exposed that monotonicity begins after the intentional 0-5% warmup; its verifier-only assertion was narrowed accordingly before any timing or score.
- [x] Prove bitwise full-step identity before 65%; at/after 65%, prove identical pre-step gradients and exact arm-specific coupled Nesterov updates from cloned fresh and preseeded-momentum fixtures.

### Milestone 3: Protect normal exposure
- [x] Run four counterbalanced complete-body accepted/candidate windows per early/hard regime after at least 20 warmups; require every CV <=5%, candidate peak <2,048 MiB, retention >=0.974644, and projected passes >=127.0 from accepted 130.304.

### Milestone 4: Run and classify the sole score
- [x] Reconfirm baseline 94.48 at `a7c42dc`, idle H20, local data, exact scope, no stale log, and passing gates; run exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Require a finite valid completion in 300.0-300.1 counted and <600 wall seconds, correct transition/evaluation cadence, 1,003,482 parameters, and no runtime fault. Record exposure; below 127 remains a valid primary score but is mechanism-inconclusive and receives no rerun.
- [x] Success is only `best_test_acc >=94.58%`; final accuracy >=94.45% and loss <=0.2456 are corroboration. A normal-exposure miss falsifies the exact 65%-anchored curve and deprioritizes nearby rephase tuning without formally rejecting all tail schedules or isolated momentum reset.

## Code Changes
- **`train.py` / `learning_rate()` only**: calculate the accepted post-warmup cosine value; return it below `MIXUP_END_FRACTION`; at or above the boundary derive the accepted transition LR, normalize remaining progress over `1-MIXUP_END_FRACTION`, and cosine to `MIN_LR`. No source outside the function changes.
- **`.autoresearch/.../experiments/039/preflight.py`**: ignored schedule/oracle/timing harness; no evaluator/test construction, production diagnostics, or `run.log` creation.

## Configuration Changes
- Pre-65% LR: returned values bitwise match accepted at sampled points, including warmup and global cosine; all non-schedule source and state remain byte-identical.
- 65% LR: unchanged `0.06123215295935604`; 100% LR: unchanged `0.002`.
- Interior tail examples: accepted/candidate LR at 75% `0.03394912/0.05008140`, 82.5% `0.01812052/0.03161608`, and 90% `0.00736409/0.01315075`; integrated tail LR area rises 39.46%.
- Model/head, batch 256, FP32, momentum 0.9 Nesterov, continuous matrix-only `5e-4` coupled decay, mixup/RandAugment timing, seed 42, loader, budget, and evaluator remain exact. Because SGD decay is coupled, the treatment increases both data-gradient motion and decay integration.

## Execution Environment
- Offline local only; no network, remote, install, W&B, GitHub, `gh`, fetch, push, or PR action.
- One NVIDIA H20, local CIFAR-10, installed `uv` environment, eight persistent workers.
- Preflight under 4 minutes; score about 346 seconds wall with a 600-second timeout; score output only in root `run.log`.

## Abort Criteria
- Abort before timing on any source/frozen-file/model/data/RNG mismatch; pre-65% LR difference; boundary discontinuity; wrong anchor/reference value; post-warmup non-monotonic or out-of-range curve; update-oracle mismatch; altered momentum state; cutoff/evaluator violation; non-finite state; or test access.
- Abort before scoring on non-finite timing, any CV >5%, retention <0.974644, projected passes <127, or candidate peak >=2,048 MiB. Print raw measurements before assertions and never rerun a stable miss.
- During score stop/classify on timeout, nonzero exit, OOM/worker/non-finite error, no output for 60 seconds, malformed summary, wrong parameter count/transitions/evaluation set, or wall >=600. Never rerun a valid result.

## Verification Protocol

### Verification Procedure

1. Run `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require 94.48 at `a7c42dc`, making threshold 94.58. Audit one idle H20, local data, experiment branch/status, exact diff, frozen `prepare.py`, and compile source plus preflight.
2. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/039/preflight.py semantics`. Guard evaluator/test construction before importing accepted/candidate sources. Require exact model/buffer bytes, parameter registration, optimizer groups/options/state, transforms/constants, construction CPU/CUDA RNG, and 1,003,482 parameters; source reconstruction must replace exactly one `learning_rate()` function.
3. Independently evaluate accepted and candidate formulas in Python float64 at 0%, 5%, `65%-1e-12`, 65%, `65%+1e-12`, 70%, 75%, 82.5%, 90%, 95%, and 100%, plus a dense grid. Require bitwise returned-value equality at sampled points below 65%; finite bounds `[0.002,0.2]`; absolute error <=`1e-12` for boundary continuity, reference values, and exact 0.002 endpoint; non-increase after the intentional 0-5% warmup with maximum positive adjacent delta <=`1e-15`; strict candidate elevation on sampled `(65%,100%)`; and absolute error <=`1e-10` from analytic tail-area ratio `1.3946300912086436`.
4. From cloned state/RNG, run production-equivalent full early mixup steps below 65%; require exact LR, inputs, mix/permutation, logits, loss, gradients, BN buffers, optimizer/parameter state, and terminal RNG. At 65%, 75%, and 90%, run hard-label arms from fresh common snapshots; require pre-step equality and independently calculate each coupled-decay Nesterov update at its prescribed LR. Repeat a representative hard point with deterministic nonempty prior momentum buffers; require exact oracle state and replay.
5. Statically and dynamically prove the first observed progress at or above 65% uses hard labels and the candidate LR computed independently at that exact pre-step progress; require exact anchor equality only for the synthetic `progress == 0.65` probe. The separately worker-gated RandAugment policy must still flip only after iterator exhaustion. Require no schedule RNG consumption and unchanged once-per-epoch evaluator/summary behavior.
6. Run `timeout 240s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/039/preflight.py timing`. Use schedule progress exactly 50% for the mixup benchmark and 75% for the hard-tail benchmark, at least 20 disposable warmups, and four paired windows of at least 50 steps with fresh equivalent fixtures per pair and counterbalanced order. Include pinned H2D, LR calculation/write, zeroing, mixup when active, forward/loss/backward/Nesterov/sync.
7. Print all windows, medians, population CVs, retention, projected passes, and candidate peak before assertions. Require CV <=0.05, peak <2,048 MiB, `retention=(0.65/c_mix+0.35/c_hard)/(0.65/a_mix+0.35/a_hard)>=0.974644`, and `130.304*retention>=127`.
8. Reaudit and execute exactly `timeout 600s uv run train.py > run.log 2>&1` once. Parse one finite summary, 300.0-300.1 counted, <600 wall, 1,003,482 parameters, exactly one mixup transition and later exhausted-iterator RandAugment transition, and evaluation epochs equal every fifth union final with no duplicate. Audit each logged LR against the candidate formula at its rounded reported progress tolerance; at the transition, use the exact observed pre-step progress formula rather than requiring exact anchor equality.
9. Record passes=`num_steps*256/50000`. Classify only by best >=94.58; report final versus 94.45, loss versus 0.2456, best-final gap, exposure, cadence, resources, timing, and final diff. Success supports only the complete fixed-seed rephase package; a valid >=127-pass miss falsifies this exact curve and operationally deprioritizes nearby tuning. It neither isolates data-gradient versus coupled-decay effects nor closes every schedule or momentum reset.

### Informational Metrics (Optional)
- `run.log`: best/final/loss, counted/wall/startup, epochs/steps/passes, VRAM, parameters, transitions/evaluation set, and representative logged LRs.
- Preflight stdout: reference/dense schedule checks, boundary jumps, tail-area ratio, cloned-step/oracle deltas, raw timing/CVs/retention/projection/peak.
