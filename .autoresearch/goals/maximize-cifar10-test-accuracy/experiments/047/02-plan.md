# Plan EXP-047: Post-GAP Feature Mixup Replacement
- **Created**: 2026-07-27

## Milestones

### Milestone 1: Implement the fixed bundled replacement
- [x] Create EXP047 branch from accepted `a7c42dc`; modify only `train.py` and freeze evaluator/`prepare.py`.
- [x] Replace early input blending with identical-draw pairing and one post-GAP/pre-MLP feature blend; preserve exact default hard/evaluation path.

### Milestone 2: Prove placement, algebra, state, and RNG
- [x] Add ignored evaluator-free `preflight.py`; prove exact state, hard-path identity, sole blend placement, no pixel mixing, feature/target alignment, and independent forward/Jacobian oracles.
- [x] Prove early/hard gradients and fresh/preseeded Nesterov updates, accepted draw trajectory, strict65% transition, and report non-gating Jensen/BN/gradient diagnostics.

### Milestone 3: Protect exposure
- [x] Run counterbalanced complete-step early/hard H20 timing with one live arm; require stable ratios, every retention>=127/130.304, median projection>=127, and peak<2,048MiB.

### Milestone 4: Sole score
- [x] Reaudit baseline94.48/source/resource/log and run exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Require valid300-second completion/cadence/transitions/1,003,482 params; success additionally requires best>=94.58 and realized passes>=127.

## Code Changes

- **`train.py` / pairing helper**: replace `mixup_batch(inputs,targets,distribution)` with `mixup_pairing(targets,distribution)` that draws the same scalar then same permutation and returns `targets`, permuted targets, scalar, and permutation without blending pixels.
- **`train.py` / `WideResNet.forward()`**: add internal default `feature_mix=None`; after final BN/ReLU/GAP/flatten and before the accepted pooled MLP, blend `out = mix*out + (1-mix)*out[permutation]` iff provided. Default path is operation-identical.
- **`train.py` / early branch**: draw pairing, call `model(inputs,feature_mix=(mix,permutation))`, and retain the exact paired CE weights/targets. Hard/evaluation calls remain `model(inputs)`.
- **ignored `experiments/047/preflight.py`**: block evaluator/test construction; load accepted source from git; prove semantic/gradient/update/RNG/timing gates and print before assertions.

## Configuration Changes

- Early interpolation site: raw input tensor -> post-GAP128-vector immediately before accepted MLP.
- Input mix and feature mix are mutually exclusive; one batch-shared Beta(0.2,0.2), ordinary permutation, paired targets,65% cutoff, clean tail, RandAugment, and all hyperparameters remain exact. State schema/construction bytes and optimizer configuration remain exact; early BN buffers, parameters, and momentum intentionally diverge under the bundled treatment.
- Interpretation is bundled clean-spatial-BN plus post-GAP interpolation; no claim isolates either component.

## Execution Environment

- Method: offline local semantic/timing gates then one local score; no network, remote, GitHub, install, W&B, or CPU loader qualification.
- Resources: one idle NVIDIA H20, local CIFAR, installed `uv`,8 accepted persistent workers.
- Estimated runtime: semantics<=180s, timing<=240s, score about345s and killed at600s.
- Log output: preflight stdout; sole project-root `run.log`, retained through analysis and removed before next experiment.
- Tool skill: `/research-execute` only.

## Abort Criteria

- Abort before timing on source/state/default-path/placement/draw/algebra/Jacobian/gradient/update/RNG/control failure; repair only a demonstrated verifier/implementation defect without changing placement, mix law, cutoff, or compound semantics.
- Abort before score on contention, nonfinite timing, arm CV>5%, ratio CV>1%, any retention<127/130.304, median projected<127, or peak>=2,048MiB.
- Abort score on timeout/nonzero/OOM/worker/nonfinite/no-output60s/malformed summary/wrong device/state/cadence/transition. A structurally valid low-exposure result is recorded once, never invalidated or rerun.

## Verification Protocol

### Verification Procedure

1. Query baseline and require94.48 at `a7c42dc`; persist HEAD/branch, current and baseline SHA-256 for `train.py`,`prepare.py`,`pyproject.toml`, exact diff scope, frozen-file diff, `git diff --check`, ignored harness, data/log/compile state, CUDA visibility/count/name, and `nvidia-smi` utilization/memory/process audits. Require EXP047 and one idle H20.
2. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/047/preflight.py semantics`; block Eval/test data and load accepted source from exact git object.
3. Diff/AST allow only the pairing helper, optional forward argument/post-GAP conditional, and early call. Require no constants/state/spatial/head/classifier/loss weights/optimizer/data/worker/schedule/evaluator/cadence changes; exactly1,003,482 params/52 parameter tensors/97 state entries.
4. Construct accepted/candidate from cloned seed42 CPU/CUDA states; require all named parameter/buffer bytes/order, pooled seed36036 matrices, optimizer membership/options/state, and post-construction RNG exact.
5. With `feature_mix=None`, require fixed FP64 CPU and FP32 CPU/CUDA logits/BN updates/losses/gradients/fresh and preseeded Nesterov updates/RNG accepted-exact in train/eval. Require hard source/default call samples no mix RNG.
6. Bind the oracle to scored tensors: temporarily wrap the candidate module's actual `F.adaptive_avg_pool2d` call to retain its returned tensor/gradient, and attach a forward pre-hook to `pooled_head[0]` for actual `z_mix`; restore the function immediately afterward. Also capture pooled-head output, refined-vector classifier input, and logits. Use a fixed coefficient0.3 and a non-self-inverse permutation containing a3-cycle; reproduce `z`, `z_mix`, `z_mix+0.1*W2*ReLU(W1*z_mix)`, logits, and correctly aligned paired CE independently within FP64 `1e-10/1e-12` and FP32 `2e-5/2e-7`. Require exactly one pre-MLP blend.
7. Prove conv1 receives original inputs and all through-GAP activations/BN updates equal an ordinary hard forward; no mixed pixels, second forward, post-MLP/logit blend, detach, or auxiliary call. Use a nonlinear fixture distinguishing pre-MLP blend from mixed head outputs.
8. With the same coefficient0.3/non-self-inverse cycle and arbitrary upstream `q`, require gradient on the retained actual pooled output to equal `lambda*q + index_add(permutation,(1-lambda)*q)` in FP64/FP32. Require finite nonzero complete early/hard gradients and independent fresh/preseeded coupled-decay Nesterov equations for all52 parameters. Separately compare expected BN running mean/variance/counter transitions and require every other buffer unchanged.
9. From cloned pre-draw CUDA state require accepted/candidate coefficient, permutation, post-draw/post-step RNG equality. Prove lambda1/identity permutation invariants, natural self/same-class handling, batch256/drop-last, strict65% predicate, one ordered RandAugment boundary, finite guard, sole backward/step, and unique every-fifth-plus-final evaluator condition.
10. Report without gating: pair cosine/distance, mixed/unmixed norm, self/same-class incidence, head Jensen gap, accepted-input-mix/candidate logits/loss deltas, grouped gradient cosines/norms, and clean/mixed-input BN deltas. Diagnostics cannot select another placement or compound treatment.
11. Run `timeout 240s uv run python .../experiments/047/preflight.py timing`. Use pinned fixtures/full production body,>=20 warm steps per arm/regime, one live GPU model, identical restored model/optimizer/input/RNG per window, and exactly two blocks `AE,CE,AH,CH,CH,AH,CE,AE`, each retained window>=50.
12. Print all16 windows, arm CVs, four early ratios, four hard ratios, four fixed-time retentions, median projection, peak. Within each eight-window block pair positions `(AE0,CE1,AH2,CH3)` and reverse `(AE7,CE6,AH5,CH4)`, yielding exactly four local retentions. Deallocate accepted/warmup models; synchronize/reset peak statistics immediately before every candidate retained window and sample immediately after, keeping the maximum. Require arm CV<=5%, ratio CV<=1%, every `(0.65/CE+0.35/CH)/(0.65/AE+0.35/AH)>=127/130.304`, median projection>=127, peak<2,048MiB.
13. Re-run/persist step1 audits, remove stale log, launch exact score once. Require exit0, CUDA, one finite summary,300.0-300.1 counted,<600 wall,1,003,482 params, correct single transitions/evaluation cadence, no errors. Retain log through analysis.
14. Compute passes=`steps*256/50000`. Structural validity is exposure-independent; success requires passes>=127 and best>=94.58. Endpoint metrics descriptive. A normal-exposure miss falsifies only exact bundled post-GAP replacement; other sites are declined, not empirically disproven.

### Informational Metrics (Optional)

- `run.log`: best/final/loss, counted/wall/startup, VRAM, epochs, steps/passes, params, transitions/evaluations.
- Preflight: oracle errors, Jensen/BN/gradient diagnostics, timing windows/CVs/retentions/projection/peak.
