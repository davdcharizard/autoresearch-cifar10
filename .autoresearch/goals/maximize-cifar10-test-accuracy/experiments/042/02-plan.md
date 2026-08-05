# Plan EXP-042: Exact-Neutral Centered Content-Attention Pooling
- **Created**: 2026-07-27

## Milestones

### Milestone 1: Implement the exact-neutral spatial pool
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-042` from accepted `a7c42dc`; modify only `train.py` and freeze `prepare.py`/evaluator.
- [x] Append one zero-initialized bias-free `Conv2d(128,1,1)` scorer under a restoring CPU RNG fork. Preserve accepted GAP and add only the centered temperature-one softmax correction before the accepted pooled MLP.

### Milestone 2: Prove identity, gradient opening, and update semantics
- [x] Add ignored evaluator-free `experiments/042/preflight.py` using exact `git show a7c42dc:train.py`; prove common state/RNG/optimizer identity, exact zero correction/default logits on CPU/CUDA, and independent nonzero-scorer pooling formulas.
- [x] Prove the float64 covariance scorer-gradient formula, accepted initial common gradients, nonzero scorer gradients in early/hard regimes, post-update nonuniform attention, and fresh/preseeded all-parameter coupled-Nesterov updates.

### Milestone 3: Protect normal exposure
- [x] Run two complete counterbalanced `A/C/C/A` cycles, yielding four retained windows per implementation per early/hard regime after at least 20 warmups; require every CV <=5%, candidate peak <2,048 MiB, retention >=0.9746439096, and projected passes >=127.0 from accepted 130.304.

### Milestone 4: Run and classify the sole score
- [x] Reconfirm baseline 94.48 at `a7c42dc`, idle H20, local data, exact scope, no stale log, and passing gates; run exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Require finite completion in 300.0-300.1 counted and <600 wall seconds, correct transitions/cadence, 1,003,610 parameters, and no runtime fault. Success is only `best_test_acc >=94.58%`; final >=94.45% and loss <=0.2456 are corroboration. A valid >=127-pass miss closes the exact treatment and immediate rescues.

## Code Changes
- **`train.py` / `WideResNet.__init__()`**: after the complete accepted model and seed-36036 pooled head are initialized, create exactly one `nn.Conv2d(widths[2], 1, 1, bias=False)` inside `torch.random.fork_rng(devices=[])` and overwrite its weight with exact zero. Add no seed, bias, buffer, temperature, gain, or positional parameter.
- **`train.py` / final pooling lines**: retain accepted `adaptive_avg_pool2d(out,1)` as `[B,128]` `mean_pooled`. Use `spatial_features=out.flatten(2)` (`[B,128,64]`), `score_logits=self.pool_score(out).flatten(1)` (`[B,64]`), `attention_delta=softmax(score_logits,dim=1)-1/score_logits.size(1)`, and `pooled_correction=torch.bmm(spatial_features,attention_delta.unsqueeze(2)).squeeze(2)` (`[B,128]`). Add correction before the exact pooled MLP/classifier. Do not replace GAP by a direct weighted reduction because reduction ordering would lose exact accepted startup.
- **`.autoresearch/.../experiments/042/preflight.py`**: ignored semantic/gradient/update/timing harness; no evaluator/test construction, tracked source mutation, `run.log`, or diagnostic-driven treatment choice.

## Configuration Changes
- Final pool: uniform global mean -> temperature-one single-query content attention expressed as accepted mean plus centered correction.
- New scorer: `Conv2d(128,1,1,bias=False)`, exactly 128 zero-initialized trainable parameters; total `1,003,482 -> 1,003,610`.
- Optimizer: generic rank-at-least-two grouping places scorer once at accepted coupled `5e-4` decay, LR, momentum 0.9, and Nesterov. Its initial decay contribution is zero; no special group.
- Everything else remains accepted: `(2,2,3)` WRN, scale-0.1 bias-free pooled head/seed 36036, batch256, FP32, 0.2-to-0.002 global cosine, alpha-0.2 batch-shared mixup and N1/M5 RandAugment through accepted 65% boundaries, seed42, loader, sole CE, evaluator, and budget.

## Execution Environment
- Method: offline local preflight and sole score; no network, remote, install, W&B, GitHub, `gh`, fetch, push, or PR action.
- Resources: one idle NVIDIA H20, local CIFAR-10, installed `uv`, eight persistent workers; H20 currently reports 0 MiB used.
- Estimated runtime: semantic <=180s, timing <=240s, score about 345s wall and always <600s.
- Log output: disposable preflight stdout; sole score only in project-root `run.log`, removed before the next experiment.
- Tool skill: `/research-execute`; no submission skill.

## Abort Criteria
- Abort before timing on any source/frozen-file, common state/byte/RNG, parameter-count/group/options, zero-weight, attention-axis/mass, exact initial correction/logit/BN evolution, covariance-gradient, common-gradient, scorer-opening, update-oracle, temporal/cadence, or finite-value failure.
- Abort before scoring on nonfinite timing, any CV >5%, retention <0.9746439096, projected passes <127, candidate peak >=2,048 MiB, or stable timing instability. Print all measurements before assertions; never use gradient/entropy/sharpness diagnostics to add a scale, temperature, seed, or cutoff.
- During score stop/classify on timeout, nonzero exit, OOM/worker/nonfinite error, no output for 60 seconds, malformed/duplicate summary, wrong parameters/transitions/evaluation set, or wall >=600. Never rerun a valid score or tune from intermediate accuracy.

## Verification Protocol

### Verification Procedure

1. Run `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require 94.48 at `a7c42dc`, threshold94.58. Audit idle H20, local data, branch/status, exact `train.py` diff, frozen `prepare.py`, no `run.log`, and compile source/preflight.
2. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/042/preflight.py semantics`. Stub `prepare.Eval`, guard test construction, and independently compile exact accepted source from git.
3. Exact source-whitelist only scorer construction and final pooling block. From cloned seed42 states require every common state key/shape/dtype/byte, post-construction CPU/CUDA RNG, transforms/constants/schedules, and temporal controls exact; require sole appended `[1,128,1,1]` zero weight and 1,003,610 parameters. Compare ordered common parameter-name subsequences per optimizer group, require no-decay identical, decay equal accepted plus exactly `pool_score.weight` once at registration position, and all group options exact.
4. On CPU/CUDA fixed inputs capture scores, attention, centered coefficients, correction, pooled/head features, logits, BN evolution, and RNG. At zero scorer require scores/centered/correction bitwise zero, attention bitwise exact `1/64`, and accepted pooled features/logits exact. Require attention mass one independently per example and no example/channel/position state.
5. On fixed nonzero scorer tensors require centered production pooling equal independent direct weighted pooling within preregistered CPU float64 `rtol=1e-10,atol=1e-12` and FP32 CPU/CUDA `rtol=2e-5,atol=2e-7`, and differ from GAP. Under a matching cyclic spatial roll, require scores/attention/centered coefficients to roll equivariantly, while mean/correction/final pool/logits remain invariant within those bounds; require no RNG.
6. In float64 use fixed upstream `g` independent of features/query and objective `(z*g).sum()`. Require query gradient equal `sum_b Cov_population(X_b)@g_b`; at zero query require feature gradients `g_b/64`. Separately test a mean-reduced CE oracle where `g_b=dL_mean/dz_b` already contains `1/B`; if deriving unreduced per-example gradients, divide their weighted mixup sum by `B`. Print covariance ranks/norms and errors first.
7. On cloned full-model early-mixup and hard-label fixtures, require zero-state logits/loss exact and map every common gradient by parameter name with identical shape/dtype/`None` status. Excluding only scorer, require `rtol=3e-3,atol=2e-4` and print maximum absolute/relative-L2 errors before assertions; scorer data gradient must be finite/nonzero. Keep zero scores/coefficients/correction, accepted pooled/logits, common state bytes, and RNG bitwise. Print scorer-gradient and update norm ratios as non-tuning diagnostics.
8. Verify every named trainable parameter and fresh/preseeded momentum-buffer existence/dtype/value using its actual group LR and `wd` (`5e-4` iff `ndim>=2`): `d=grad+wd*p0`; fresh `b1=d`, preseeded `b1=0.9*b0+d`; `p1=p0-lr*(d+0.9*b1)`. Require non-parameter BN buffers evolve only through forward, not SGD, and replay state/RNG. On the same fixed pre-update input after the independently predicted full update, separately report early/hard score std, entropy/effective sites/max weight and require finite nonuniform attention.
9. Run `timeout 240s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/042/preflight.py timing`. For each regime warm each implementation >=20 disposable steps, then use two `A/C/C/A` cycles. Each of four retained samples per implementation is mean seconds/step over a >=50-step complete window; restore the same fixture before every sample and reset/synchronize peak accounting around candidate windows. Include pinned H2D, LR writes, zeroing, mixup, full forward/loss/backward/Nesterov/finite guard/sync.
10. Print four window means, median and population CV over them, peak, retention, and projection before assertions. Use only four-sample medians in `retention=(0.65/c_mix+0.35/c_hard)/(0.65/a_mix+0.35/a_hard)`; require CV<=0.05, peak<2,048MiB, retention>=0.9746439096, and `130.304*retention>=127`.
11. Reaudit and execute exactly `timeout 600s uv run train.py > run.log 2>&1` once. Parse one finite summary, 300.0-300.1 counted, <600 wall, 1,003,610 params, exactly one mixup and later exhausted-iterator RandAugment transition, and every-fifth union final evaluations without duplicate.
12. Record passes=`num_steps*256/50000`. Necessary success requires completion and best>=94.58. Report final vs94.45, loss vs0.2456, gap, exposure, cadence/resources/timing/diff. Success supports only the complete content pool; a valid >=127-pass miss evidentially rejects only the exact zero-started, temperature-one, one-query treatment. Temperature/init/scale/query-count/cutoff variants are declined as preregistered post-result search policy, not experimentally rejected.

### Informational Metrics (Optional)
- `run.log`: peak VRAM, final accuracy/loss, counted/wall/startup, epochs, steps/passes, parameters, transitions, and evaluations.
- Preflight stdout: identity/gradient/update errors, query gradient/update norms, post-update score std, entropy/effective sites/max weight, timing windows/CVs/retention/projection/peak.
