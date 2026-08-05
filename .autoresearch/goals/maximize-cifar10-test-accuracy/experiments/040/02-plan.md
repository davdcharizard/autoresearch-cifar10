# Plan EXP-040: Equalize Effective Classifier Row Norms
- **Created**: 2026-07-27

## Milestones

### Milestone 1: Implement the exact forward reparameterization
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-040` from accepted `a7c42dc`; modify only `train.py` and freeze `prepare.py`/evaluator.
- [x] Replace only the final affine call with differentiable `W_eff[i] = W[i]/||W[i]|| * ||W||_F/sqrt(10)` and `F.linear`; retain raw weight/bias objects, initialization, parameter count, optimizer groups, and every accepted training choice.

### Milestone 2: Prove geometry, gradient, and update semantics
- [x] Add ignored evaluator-free `experiments/040/preflight.py` using independent `git show a7c42dc:train.py`; prove exact source scope, initial state/RNG/optimizer identity, finite nonzero rows, and row/Frobenius/direction invariants on float64/CPU-FP32/CUDA-FP32 fixtures.
- [x] Verify candidate logits and raw-weight gradients against independent differentiable and analytic formulas, run float64 gradcheck, then prove fresh and preseeded coupled-Nesterov updates plus deterministic early/hard replay.

### Milestone 3: Protect normal exposure
- [x] Run four counterbalanced complete-body accepted/candidate windows per early/hard regime after at least 20 warmups; require every CV <=5%, candidate peak <2,048 MiB, retention >=0.974644, and projected passes >=127.0 from accepted 130.304.

### Milestone 4: Run and classify the sole score
- [x] Reconfirm baseline 94.48 at `a7c42dc`, idle H20, local data, exact scope, no stale log, and passing gates; run exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Require a finite valid completion in 300.0-300.1 counted and <600 wall seconds, correct transitions/evaluation cadence, 1,003,482 parameters, and no runtime fault. Record exposure; below 127 remains a valid primary score but is mechanism-inconclusive and receives no rerun.
- [x] Success is only `best_test_acc >=94.58%`; final accuracy >=94.45% and loss <=0.2456 are corroboration. A normal-exposure miss closes only this exact differentiable RMS equal-row map and algebraic equivalents.

## Code Changes
- **`train.py` / final four lines of `WideResNet.forward()` only**: after the accepted pooled residual feature, compute raw row norms, differentiable RMS row norm `||W||_F/sqrt(C)`, effective equal-norm rows, then call `F.linear(out, effective_weight, self.fc.bias)`. Do not add epsilon/clamp, temperature/gain, feature normalization, bias change, data mutation, or post-step projection.
- **`.autoresearch/.../experiments/040/preflight.py`**: ignored geometry/gradient/oracle/timing harness; no evaluator/test construction, production diagnostics, or `run.log` creation.

## Configuration Changes
- Effective class-vector radii: raw unequal norms -> common differentiable RMS raw-row norm at every forward.
- Instantaneous effective/raw classifier Frobenius norms: equal by construction; accepted initialization/state bytes remain exact, but the learned scale trajectory is not claimed to remain accepted because gradients change.
- No new parameter or scalar. Accepted model/head, batch 256, FP32, LR curve/floor, momentum 0.9 Nesterov, continuous matrix-only `5e-4` decay, mixup/RandAugment timing, seed 42, loader, budget, and evaluator remain exact.

## Execution Environment
- Offline local only; no network, remote, install, W&B, GitHub, `gh`, fetch, push, or PR action.
- One NVIDIA H20, local CIFAR-10, installed `uv` environment, eight persistent workers.
- Semantic preflight under its 180-second timeout and timing preflight under its 240-second timeout; score about 346 seconds wall with a 600-second timeout; score output only in root `run.log`.

## Abort Criteria
- Abort before timing on any source/frozen-file/model/data/RNG mismatch; changed raw parameter/buffer bytes; wrong optimizer membership/options; non-finite or row norm <=`1e-6` in sampled fixed fixtures; row/Frobenius/direction invariant failure; logit/gradient/gradcheck/update-oracle mismatch; extra RNG; evaluator/cutoff violation; or non-finite state. Fixed fixtures do not prove future rows stay nonzero; the scored finite-loss guard remains binding.
- Abort before scoring on non-finite timing, any CV >5%, retention <0.974644, projected passes <127, or candidate peak >=2,048 MiB. Print raw measurements before assertions; never rerun a stable miss or replace the map with projection/caching.
- During score stop/classify on timeout, nonzero exit, OOM/worker/non-finite error, no output for 60 seconds, malformed summary, wrong parameter count/transitions/evaluation set, or wall >=600. Never rerun a valid result.

## Verification Protocol

### Verification Procedure

1. Run `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require 94.48 at `a7c42dc`, making threshold 94.58. Audit one idle H20, local data, experiment branch/status, exact diff, frozen `prepare.py`, and compile source plus preflight.
2. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/040/preflight.py semantics`. Guard evaluator/test construction before importing candidate and an independently compiled exact `git show a7c42dc:train.py` accepted module. Require byte-identical parameter/buffer state, registration, 1,003,482 parameters, optimizer groups/options, transforms/constants, schedule/transitions, and post-construction CPU/CUDA RNG. Separately audit textual scope by replacing exactly the candidate final-classifier block with the accepted block.
3. Independently calculate effective weights on fixed CPU float64, CPU float32, and CUDA float32 tensors. Require each effective row norm equal the independent RMS raw-row norm, effective/raw Frobenius equality, cosine-one directions, unchanged bias, finite values, and minimum raw row norm >`1e-6`, using `rtol=1e-6, atol=1e-7` for FP32 and `rtol=1e-10, atol=1e-12` for float64.
4. Reproduce accepted-seed characterization: raw row CV near `0.0696447`, max/min near `1.2724986`, relative effective/raw weight delta near `0.0695184`, effective Frobenius near `4.5009542`, and nonzero initial full-input logit perturbation. These are fixed diagnostics, not evidence or tunable gates beyond detecting wrong implementation.
5. On independent pooled-feature and full-model fixtures, require production logits equal `F.linear(z, s*W/r, bias)` with separately ordered FP32 norm arithmetic under `rtol=2e-5, atol=2e-7` (passing measured maximum absolute difference `4.77e-7`, governed by combined relative/absolute bounds), while accepted/candidate pre-classifier pooled features are bitwise equal. Verify positive global scaling equivariance; for isolated-row rescaling require multiplier `a>0`, preserved row direction, and effective-weight changes only through shared RMS scale; require no RNG consumption.
6. Run `torch.autograd.gradcheck` in float64 for raw weight, pooled features, and bias. With a fixed upstream tensor `q` independent of `W`, use scalar objective `(W_eff*q).sum()` and verify `dL/dw_i = (s/r_i)(I-n_i n_i^T)q_i + w_i*A/(C*s)`. If `q` is obtained from another loss, detach it before this oracle. Also compare a separately coded differentiable formula against production gradients. Print maximum errors, tangential/radial components, row norms, and invariants before assertions.
7. From fresh common accepted/candidate snapshots, verify each arm's complete early-mixup and hard-label gradient/update against its own independent forward and coupled-decay Nesterov oracle. Repeat with deterministic nonempty prior momentum buffers; require finite exact state within declared tolerance and bitwise candidate replay from restored state/RNG. Only the instantaneous `weight_decay * W` contribution is purely radial; historical Nesterov buffers need not align with current rows. Do not claim accepted/candidate backbone gradients remain equal after transformed logits.
8. Run `timeout 240s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/040/preflight.py timing`. Use at least 20 disposable warmups and four paired windows of at least 50 steps with fresh equivalent fixtures and counterbalanced order for early mixup and hard labels. Include pinned H2D, LR writes, zeroing, full transformed forward/loss/backward, Nesterov, finite guard, and sync.
9. Print all windows, medians, population CVs, retention, projected passes, and candidate peak before assertions. Require CV <=0.05, peak <2,048 MiB, `retention=(0.65/c_mix+0.35/c_hard)/(0.65/a_mix+0.35/a_hard)>=0.974644`, and `130.304*retention>=127`.
10. Reaudit and execute exactly `timeout 600s uv run train.py > run.log 2>&1` once. Parse one finite summary, 300.0-300.1 counted, <600 wall, 1,003,482 parameters, exactly one mixup and later exhausted-iterator RandAugment transition, and every-fifth union final evaluations with no duplicate.
11. Record passes=`num_steps*256/50000`. Classify only by best >=94.58; report final versus 94.45, loss versus 0.2456, best-final gap, exposure, cadence, resources, timing, and final diff. Success supports the complete fixed-seed map without distinguishing radius removal from conditioning/coupling. A valid >=127-pass miss closes only the exact map and algebraic equivalents, not learned gain, feature normalization, detached scales, bias removal, angular penalties, or projection.

### Informational Metrics (Optional)
- `run.log`: best/final/loss, counted/wall/startup, epochs/steps/passes, VRAM, parameters, transitions/evaluation set.
- Preflight stdout: raw/effective norm geometry, logit perturbation, analytic/autograd errors, gradient decomposition, update oracles, raw timing/CVs/retention/projection/peak.
