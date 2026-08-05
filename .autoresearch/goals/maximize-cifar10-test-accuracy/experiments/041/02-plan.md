# Plan EXP-041: Training-Only Direct-Path Auxiliary Cross-Entropy
- **Created**: 2026-07-27

## Milestones

### Milestone 1: Implement the exact dual-path training objective
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-041` from accepted `a7c42dc`; modify only `train.py` and freeze `prepare.py`/evaluator.
- [x] Add an explicit opt-in dual-logit return to `WideResNet.forward()` and use the exact always-on 90/10 main/direct CE objective in both accepted mixup and hard-label branches. Preserve the default/evaluator forward and every accepted parameter, state, data, schedule, RNG, and optimization choice.

### Milestone 2: Prove inference, loss, gradient, and update semantics
- [x] Add ignored evaluator-free `experiments/041/preflight.py` using independent `git show a7c42dc:train.py`; prove exact source/state/RNG/optimizer/default-inference identity and independent dual-logit/loss formulas.
- [x] Prove mixup target reuse, analytic gradient decomposition, fresh and preseeded coupled-Nesterov updates, temporal controls, and deterministic replay for early and hard regimes.

### Milestone 3: Protect normal exposure
- [x] Run two complete counterbalanced `A/C/C/A` cycles, yielding four retained windows per implementation per early/hard regime after at least 20 warmups; require every CV <=5%, candidate peak <2,048 MiB, retention >=0.9746439096, and projected passes >=127.0 from accepted 130.304.

### Milestone 4: Run and classify the sole score
- [x] Reconfirm baseline 94.48 at `a7c42dc`, idle H20, local data, exact source scope, no stale log, and passing gates; run exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Require a finite valid completion in 300.0-300.1 counted and <600 wall seconds, correct transitions/evaluation cadence, 1,003,482 parameters, and no runtime fault. Success is only `best_test_acc >=94.58%`; final accuracy >=94.45% and loss <=0.2456 are corroboration. A valid >=127-pass miss closes the exact objective and immediate rescues.

## Code Changes
- **`train.py` / `WideResNet.forward()`**: add default-false `return_direct_logits`. Compute accepted main logits first as `self.fc(out + 0.1 * self.pooled_head(out))`; on opt-in only, compute and return `(main_logits, self.fc(out))`. Default must return the same tensor through one classifier invocation, with no `self.training`-dependent return type or evaluation work.
- **`train.py` / early mixup loss**: call dual mode once on the same accepted `mixed_inputs`. Compute accepted main mixed-target CE and the analogous direct CE from the same `mix`, `targets_a`, and `targets_b`; combine exactly `(1 - POOLED_HEAD_SCALE) * main_loss + POOLED_HEAD_SCALE * direct_loss`. Do not redraw, detach, fuse CE arithmetic, or change accepted target ordering.
- **`train.py` / hard-label loss**: call dual mode once on accepted clean inputs; combine main/direct hard CE with the same exact 0.9/0.1 convex formula. Keep the auxiliary active for the entire run; do not add a cutoff, parameter, state, diagnostic, or evaluator path.
- **`.autoresearch/.../experiments/041/preflight.py`**: ignored semantic/gradient/update/timing harness. It may not construct the real evaluator/test set, modify tracked source, create `run.log`, or choose treatment settings from diagnostics.

## Configuration Changes
- Training objective: accepted main CE -> exact `0.9 * main CE + 0.1 * direct CE`, where `0.1` is the existing `POOLED_HEAD_SCALE`; no new hyperparameter.
- Training model output: one main tensor -> explicit opt-in `(main, direct)` tuple; default/inference output remains one accepted main tensor.
- No parameter/configuration changes: retain 1,003,482 parameters, `(2,2,3)` WRN, scale-0.1 bias-free pooled head and seed 36036, batch 256, FP32, LR 0.2 to floor 0.002, momentum 0.9 Nesterov, matrix-only `5e-4` decay, alpha-0.2 batch-shared mixup and N1/M5 RandAugment through their accepted 65% boundaries, seed 42, loader, budget, and evaluator.
- Interpretive confound is preregistered: pooled-head data gradients become 0.9 of main-only while coupled decay is unchanged, raising decay/data ratio 11.1%. Do not compensate with loss rescaling or optimizer exceptions.

## Execution Environment
- Method: offline local preflight and sole score; no network, remote, install, W&B, GitHub, `gh`, fetch, push, or PR action.
- Resources: one idle NVIDIA H20, local CIFAR-10, installed `uv` environment, eight persistent workers; current H20 check reports 0 MiB used before implementation.
- Estimated runtime: semantic preflight <=180 seconds, timing preflight <=240 seconds, sole score about 345 seconds wall and always <600 seconds; analysis follows locally.
- Log output: preflight stdout for disposable diagnostics; sole score only in project-root `run.log`, removed before the next experiment.
- Tool skill: `/research-execute`; no external submission skill.

## Abort Criteria
- Abort before timing on any source/frozen-file mismatch; changed parameter/buffer bytes, state keys, count, optimizer grouping/options, or construction RNG; non-bitwise default inference; wrong dual-logit call order/formula; extra RNG/state mutation; incorrect 90/10 loss or mixup-target reuse; gradient/update oracle mismatch; temporal/evaluator violation; or nonfinite value.
- Abort before scoring on nonfinite timing, any CV >5%, retention <0.9746439096, projected passes <127, candidate peak >=2,048 MiB, or an unstable/repeated timing request. Print raw measurements before assertions.
- During score stop/classify on timeout, nonzero exit, OOM/worker/nonfinite error, no output for 60 seconds, malformed/duplicate summary, wrong parameter count/transitions/evaluation set, or wall >=600. Never rerun a valid score or tune coefficient/cutoff/detach/head after observing diagnostics or metrics.

## Verification Protocol

### Verification Procedure

1. Run `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require 94.48 at `a7c42dc`, making threshold 94.58. Audit one idle H20, local data, experiment branch/status, exact `train.py` diff, frozen `prepare.py`, no `run.log`, and compile source/preflight.
2. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/041/preflight.py semantics`. Stub `prepare.Eval` and guard evaluator/test construction before independently compiling candidate and exact `git show a7c42dc:train.py` accepted modules.
3. From cloned seed-42 CPU/CUDA RNG states, require identical state keys/shapes/dtypes/bytes, 1,003,482 parameters, post-construction RNG, optimizer membership/order/options, transforms/constants, schedule, and transitions. Compare text and AST against exact `git show a7c42dc:train.py`, whitelisting only the `forward` signature, pooled/refined/main/direct return block, and two inline training-loss branches; every other production token remains accepted.
4. On fixed CPU/CUDA fixtures, require default candidate `model(x)` to return one `[B,10]` tensor bitwise equal to the independently loaded accepted module with exactly one `fc` invocation. In dual mode capture ordered `fc` hook inputs: call 1 must be bitwise independent `z + 0.1*h(z)`, call 2 bitwise raw `z`; the returned main must equal default and the returned pair must equal the two independently evaluated affine calls. Require accepted BN-buffer evolution and no extra CPU/CUDA RNG.
5. Independently reproduce early mixup with one fixed lambda/permutation and hard labels. Treat the numerical fixture as a formula oracle: require main losses bitwise equal to accepted before blending, direct losses finite/nondegenerate, and the explicit nested 90/10 result exact within declared tolerance. Separately AST/source-audit the scored inline branches: exactly one `mixup_batch`; one dual forward on its `mixed_inputs`; the same `mix`, `targets_a`, and `targets_b` in both path losses; then exactly `(1-s)*main_loss + s*direct_loss`. Require the hard branch to use one dual forward on `inputs`, the same `targets` twice, and the same blend. No detach, redraw, second mixture, or alternate operation order is allowed.
6. For each early/hard regime, obtain loss gradients from three cloned models at identical parameter and BN state: main-only, direct-only, and combined. Before adding decay, check every pooled-head tensor has direct-only gradient `None` or exact zero and combined gradient equal to `0.9*g_main`; check every other trainable tensor against `0.9*g_main + 0.1*g_direct` using printed preregistered absolute/relative FP32 tolerances. Print CE ratios, main/direct agreement, and gradient cosines/norms before assertions as non-tuning diagnostics.
7. For both regimes, verify every parameter and resulting momentum buffer under fresh and deterministic preseeded momentum. Independently compute `d=grad+wd*p0`; fresh `b1=d`, preseeded `b1=0.9*b0+d`; and Nesterov `p1=p0-lr*(d+0.9*b1)`, with `wd=5e-4` only for rank-at-least-two tensors and zero otherwise. Restore state/RNG and require bitwise candidate replay, finite state, unchanged data randomness, and exact one-way 65% controls.
8. Run `timeout 240s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/041/preflight.py timing`. For each early/hard regime, use at least 20 warmups then two complete `A/C/C/A` cycles of windows >=50 steps, yielding four accepted and four candidate retained windows. Restore equivalent model/optimizer fixtures for each paired window, reset peak-memory accounting, and include H2D, LR writes, zeroing, full dual forward/loss/backward, Nesterov, finite guard, and synchronization.
9. Print all windows, median seconds per complete step (`a_mix`, `c_mix`, `a_hard`, `c_hard`), population CVs, retention, projected passes, and candidate peak before assertions. Require CV <=0.05, peak <2,048 MiB, `retention=(0.65/c_mix+0.35/c_hard)/(0.65/a_mix+0.35/a_hard)>=0.9746439096`, and `130.304*retention>=127`; compute retention only from these reciprocal medians.
10. Reaudit and execute exactly `timeout 600s uv run train.py > run.log 2>&1` once. Parse one finite summary, 300.0-300.1 counted, <600 wall, 1,003,482 parameters, exactly one mixup and later exhausted-iterator RandAugment transition, and every-fifth union final evaluations with no duplicate.
11. Record passes=`num_steps*256/50000`. Necessary success requires completion and best >=94.58. Report final versus 94.45, loss versus 0.2456, best-final gap, exposure, cadence, resources, timing, and final diff. A valid >=127-pass miss rejects only the exact always-on shared-classifier 90/10 objective and immediate coefficient/cutoff/detach/separate-head/distillation/head-scale rescues; success supports the complete coupled objective without proving raw-feature collapse or isolating its gradient mechanisms.

### Informational Metrics (Optional)
- `run.log`: `peak_vram_mb`, final accuracy/loss, counted/wall/startup time, epochs, steps/passes, parameter count, transitions, and evaluation set.
- Preflight stdout: main/direct losses and agreement, gradient norms/cosines/decomposition errors, update-oracle errors, timing windows/CVs/retention/projection/peak.
