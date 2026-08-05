# Plan EXP-038: Double Only Terminal Classifier Decay
- **Created**: 2026-07-27

## Milestones

### Milestone 1: Implement the exact three-way optimizer allocation
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-038` from accepted `a7c42dc`; modify only `train.py` and freeze `prepare.py`/evaluator.
- [x] Add `CLASSIFIER_WEIGHT_DECAY = 1e-3`; derive three optimizer groups from `model.named_parameters()` so exactly `fc.weight` receives `1e-3`, every other trainable matrix receives accepted `5e-4`, and every rank-below-2 tensor receives zero decay.

### Milestone 2: Prove allocation and update semantics
- [x] Add ignored evaluator-free `experiments/038/preflight.py` using independent `git show a7c42dc:train.py`; prove exact model/data/RNG/source identity, 1,003,482 parameters, three disjoint exhaustive groups, and exact 999,856/1,280/2,346 element counts.
- [x] Prove accepted/candidate pre-step identity for mixup and hard paths, then verify first and preseeded-momentum Nesterov updates against independent coupled-decay oracles: only `fc.weight` may differ directly, with finite states and unchanged RNG.

### Milestone 3: Protect the accepted exposure regime
- [x] Run four counterbalanced complete-body accepted/candidate windows per early/hard regime after at least 20 warmups; require every CV <=5%, candidate peak <2,048 MiB, retention >=0.974644, and projected passes >=127.0 from accepted 130.304.

### Milestone 4: Run and classify the sole score
- [x] Reconfirm baseline 94.48 at `a7c42dc`, idle H20, local data, exact scope, no stale log, and passing gates; run exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Require a finite valid completion in 300.0-300.1 counted and <600 wall seconds, correct transitions/cadence, 1,003,482 parameters, and no runtime fault. Record exposure; a result below 127 remains a valid primary-metric observation but is mechanism-inconclusive and must not be rerun.
- [x] Success is only `best_test_acc >=94.58%`; final accuracy >=94.45% and loss <=0.2456 are corroboration. A valid >=127-pass miss rejects the two tested one-sided classifier-decay perturbations (`0` and `1e-3`) and deprioritizes nearby static tuning; intermediate values and schedules remain formally untested.

## Code Changes
- **`train.py` / optimizer allocation only**: add the classifier-specific decay constant; enumerate `model.named_parameters()` into (1) all trainable `ndim>=2` tensors except exactly `fc.weight`, (2) exactly `fc.weight`, and (3) all trainable `ndim<2` tensors. Preserve every SGD option and all other source behavior.
- **`.autoresearch/.../experiments/038/preflight.py`**: ignored semantic/oracle/timing harness; no evaluator/test access, production diagnostics, or `run.log` creation.

## Configuration Changes
- `fc.weight` decay: `5e-4 -> 1e-3` continuously for all training; 1,280 elements move to a dedicated group.
- Representation matrix group: 999,856 elements including every convolution and both pooled-head matrices at `5e-4`. Classifier group: 1,280 elements at `1e-3`. Zero-decay group: 2,346 rank-below-2 elements.
- The doubled value is the symmetric opposite bracket to EXP037's `0.0` around accepted `5e-4`; it tests direction without assuming the zero-decay miss proves monotonicity.
- Model/head, batch 256, FP32, LR `0.2 -> 0.002`, momentum 0.9 Nesterov, mixup/RandAugment timing, seed 42, loader, budget, and evaluator remain exact.

## Execution Environment
- Offline local only; no network, remote, install, W&B, GitHub, `gh`, fetch, push, or PR action.
- One NVIDIA H20, local CIFAR-10, installed environment, eight persistent workers.
- Preflight under 4 minutes; score about 346 seconds wall with a 600-second timeout; score output only in root `run.log`.

## Abort Criteria
- Abort before timing on any source/frozen-file/model/data/RNG mismatch; wrong group count/membership/options; representation or pooled-head decay change; pre-step mismatch; oracle/update mismatch beyond `fc.weight`; non-finite state; cutoff/evaluator violation; or test access.
- Abort before scoring on non-finite timing, CV >5%, retention <0.974644, projected passes <127, or candidate peak >=2,048 MiB. Print raw values before assertions; never rerun a stable miss.
- During score stop/classify on timeout, nonzero exit, OOM/worker/non-finite error, no output for 60 seconds, malformed summary, wrong parameters/transitions/evaluation set, or wall >=600. Never rerun a valid result.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline`; require 94.48 at `a7c42dc`, making threshold 94.58. Audit one idle H20, local data, branch/status, exact diff, frozen `prepare.py`, and compile source plus preflight.
2. Run `timeout 180s uv run python .../experiments/038/preflight.py semantics`. Guard evaluator/test construction before importing accepted/candidate sources. Require byte-equal model/buffer state, registration order, post-construction CPU/CUDA RNG, 1,003,482 parameters, and the exact constant-plus-three-comprehension source diff.
3. Enumerate groups by name and identity. Require each trainable tensor exactly once; representation decay exactly all `ndim>=2` except `fc.weight` with 999,856 elements at `5e-4`; classifier exactly `fc.weight` with 1,280 elements at `1e-3`; zero decay exactly all `ndim<2` with 2,346 elements. Require both pooled-head matrices decayed at `5e-4` and every other SGD setting accepted.
4. From cloned state/RNG run production-equivalent early and hard forward/backward without stepping; require exact inputs, lambda/permutation when active, logits, loss, gradients, BN buffers, and RNG. At a fixed production LR, independently calculate coupled-decay Nesterov first-step oracles; require all non-`fc.weight` parameters/buffers equal, each arm's `fc.weight` matches its full-update oracle, the cross-arm difference is finite/nonzero, and the candidate's decay contribution is exactly twice the accepted contribution.
5. From a fresh common model snapshot, seed identical deterministic finite nonempty prior momentum buffers and run one hard step per arm. Require each arm's buffer equal `0.9*b_prev+d_t`, parameter direction equal `d_t+0.9*b_t`, and replay reproducibility. Cross-arm pre-step gradients are equal in this fresh fixture; do not continue from the divergent first-step fixture or claim later-trajectory equality.
6. Run `timeout 240s uv run python .../experiments/038/preflight.py timing`. Use at least 20 disposable warmups and four paired windows of at least 50 steps, with fresh equivalent model/BN/optimizer/RNG/input fixtures per pair and counterbalanced order. Include pinned H2D, LR writes, zeroing, mixup when active, forward/loss/backward/Nesterov/sync.
7. Print all windows, medians, population CVs, retention, projected passes, and candidate peak before assertions. Require CV <=0.05, peak <2,048 MiB, `retention=(0.65/c_mix+0.35/c_hard)/(0.65/a_mix+0.35/a_hard)>=0.974644`, and `130.304*retention>=127`.
8. Reaudit and execute the sole exact score command. Parse one finite summary, 300.0-300.1 counted, <600 wall, 1,003,482 parameters, one mixup and exhausted-iterator RandAugment transition with source-valid boundary ordering, and evaluation epochs equal every fifth union final with no duplicate.
9. Record passes=`num_steps*256/50000`. Classify only by best >=94.58; report final versus 94.45, loss versus 0.2456, best-final gap, exposure, cadence, resources, timing, and final diff. A valid miss at >=127 passes rejects the two tested one-sided perturbations and deprioritizes nearby static classifier-decay tuning while leaving intermediate values and schedules formally untested. A below-127 completion is still a valid score but supports no mechanism-level decay conclusion and receives no rerun.

### Informational Metrics (Optional)
- `run.log`: best/final/loss, counted/wall/startup, epochs/steps/passes, VRAM, parameters, transitions/evaluation set.
- Preflight stdout: exact membership/counts, first/preseeded-step oracle deltas, raw timing/CVs/retention/projection/peak.
