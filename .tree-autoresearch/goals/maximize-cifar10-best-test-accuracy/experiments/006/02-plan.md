# Plan EXP-006: Shared-Budget CutMix and Manifold Mixup
- **Created**: 2026-08-05

## Goal and Fixed Reference

- Parent node: EXP-004 (`1a8d0de`), `best_test_acc=95.40%`.
- Primary acceptance threshold: `best_test_acc >= 95.50%`.
- Chosen mechanism: preserve the parent's early `p=0.5` mixed-batch gate, route 75% of selected batches to unchanged CutMix, and route 25% to `Beta(2,2)` manifold mixup split equally across the first two WRN stage boundaries.
- Expected range: 95.55-95.80%, or +0.15 to +0.40 points, after heavily discounting longer and weaker source recipes.
- Attribution: any result supports or rejects the combined asymmetric CutMix/manifold policy. It cannot isolate manifold mixup from the 25% reduction in selected CutMix exposure or identify one boundary as causal.

## Milestones

### Milestone 1: Implement deterministic shared-budget routing
- [x] Add fixed manifold policy constants, a private seed-43 CPU generator, and private seed-44 CPU/CUDA generators in `train.py` only.
- [x] Keep the parent's gate draw on the existing seed-42 CutMix CPU generator. Every selected batch must consume the exact parent CutMix lambda, center, and CUDA permutation draws before private mode routing, even if that spec is discarded for manifold mixup.
- [x] Route selected batches using one private mode draw: 75% CutMix, 25% manifold. Split manifold batches uniformly between boundaries after blocks 2 and 4.
- [x] Implement exact `Beta(2,2)` sampling from four private CPU uniforms using two integer-shape Gamma(2,1) draws; use one private CUDA permutation for manifold targets/features.
- [x] Add counters for total early eligible/selected/clean batches, CutMix batches, manifold batches, boundaries 2/4, discarded parent CutMix specs, mean manifold lambda, and mean `min(lambda, 1-lambda)`.
- [x] Verification: a 100,000-step deterministic policy simulation must reproduce the preregistered exact counts and moments, exact generator advancement, and bitwise parent CutMix-generator parity for every clean/selected decision in the shared prefix.

### Milestone 2: Add one-pass hidden-representation mixing
- [x] Extend `PreActWideResNet.forward` with optional `mix_boundary`, `mix_permutation`, and `mix_lambda` arguments whose defaults leave clean, CutMix, SAM, and evaluator calls bit-identical.
- [x] On a manifold batch, interpolate once and out of place immediately after completed block 2 or 4, restore channels-last contiguity, then continue the remaining blocks normally.
- [x] Use paired targets from the same manifold permutation and the sampled lambda in the existing two-term cross-entropy. Never area-correct a manifold coefficient.
- [x] Assert all hidden-mix arguments are present together, the boundary is exactly 2 or 4, permutation device/shape is correct, lambda lies strictly in `(0,1)`, and mixing executes exactly once.
- [x] Preserve the existing DataLoader, 256 independent images per batch, model parameters, global RNG stream, CutMix geometry, drop-path calls, optimizer, and late SAM implementation.
- [x] Verification: default-forward parity, exact boundary/label math, two-source gradients, no aliasing, CutMix regression, global RNG isolation, and BF16/channels-last GPU integration smokes must pass.

### Milestone 3: Run one full GPU-0 experiment
- [x] Confirm physical GPU 0 is an NVIDIA H20 with approximately 98 GB memory, the branch is EXP-006 at parent `1a8d0de`, only `train.py` differs, and no stale `run.log` exists.
- [x] Launch exactly once with fixed seed 42, a 600-second outer timeout, and complete stdout/stderr redirection to `run.log`.
- [x] Monitor only for process health and abort criteria; do not prune, retry, or adjust from intermediate test accuracy.
- [x] Verification: require exit 0, complete summary, approximately 300 charged seconds, total runtime below 600 seconds, at least 24,000 optimizer steps, one evaluation per epoch, and an auditable policy/SAM trace.

### Milestone 4: Verify and record the preregistered result
- [x] Check policy frequencies and generator-parity counters before reading the primary metric.
- [x] Compare `best_test_acc` once with 95.50%; a miss is no-improvement without a rerun or parameter change.
- [x] Report best/final accuracy, final loss versus 0.1654, steps versus 25,560, runtime, VRAM, evaluation count, all policy counts, mean manifold lambda, and unchanged model size.
- [x] Attribute the result only to the combined hybrid policy and remove `run.log` after execution and analysis evidence are recorded.

## Code Changes

- **`train.py` only**:
  - Add `MANIFOLD_SHARE=0.25`, `MANIFOLD_ALPHA=2.0`, `MANIFOLD_BOUNDARIES=(2,4)`, `POLICY_SEED=43`, and `MANIFOLD_SEED=44`.
  - Refactor the existing CutMix random draws into a helper returning nominal lambda, center, and CUDA permutation. Keep their order exactly: selected-gate CPU draw, lambda CPU draw, center-x CPU draw, center-y CPU draw, CUDA permutation.
  - Draw the selected gate from the existing CutMix CPU generator exactly as EXP-004. On every selected batch, draw the full CutMix spec before the private mode. Apply it unchanged for CutMix mode; discard it for manifold mode. This preserves the seed-42 CutMix CPU/CUDA states and selected/clean sequence relative to the parent for any shared step prefix; it does not assert equal total steps at a time-based cutoff.
  - Add a private seed-43 CPU generator. Consume one uniform mode draw per selected batch; if manifold is chosen, consume one `randint(2)` boundary draw. The fixed threshold is `mode_draw < 0.25` for manifold.
  - Add private seed-44 CPU/CUDA generators. Sample manifold lambda from four CPU uniforms via `g1=-log(u1)-log(u2)`, `g2=-log(u3)-log(u4)`, `lambda=g1/(g1+g2)`; clamp uniforms only to a fixed positive numerical floor before log. Draw one CUDA permutation.
  - Extend the model forward signature with optional mixing arguments. After block 2 or 4, compute `paired = out[permutation]` and `out = lambda*out + (1-lambda)*paired` out of place, then make and assert the result contiguous in channels-last format before the next block. Defaults must execute no mix branch and consume no RNG.
  - For manifold mode, set `targets_b=targets[permutation]`, set the loss coefficient to the exact sampled lambda, and pass the same boundary/permutation/lambda to the sole model forward. For CutMix, keep current patch mutation and clipped-area coefficient. Clean batches remain unchanged.
  - Generalize the existing SAM/mixed-target overlap assertion. All mixing ends at progress 0.75, so both SAM passes retain default model calls, hard labels, RNG replay, BatchNorm suppression, exact restore, and one Nesterov update.
  - Add startup config and final `mix_policy:` / `manifold:` audit lines, including both lambda mean and mean `min(lambda, 1-lambda)`. Avoid per-step device synchronizations; manifold lambda and policy statistics are CPU values.
  - Do not change architecture, parameter initialization, DataLoader, transforms, batch size, parent gate probability/cutoff, CutMix alpha/geometry/seeds, drop path, optimizer, LR schedule, SAM rho/start/cadence, evaluator, timing boundaries, or required final summary keys.

## Configuration Changes

- Selected early mix routing: `100% CutMix` -> `75% CutMix / 25% manifold mixup`.
- Marginal early probabilities: `clean=0.50, CutMix=0.50` -> `clean=0.50, CutMix=0.375, manifold@block2=0.0625, manifold@block4=0.0625`.
- `MANIFOLD_ALPHA`: absent -> `2.0` (main paper CIFAR setting).
- `MANIFOLD_BOUNDARIES`: absent -> `(2,4)` (first two complete WRN stages; no classifier/final-stage mixing).
- Private generator seeds: absent -> mode/boundary 43, manifold lambda/permutation 44. These are fixed namespaces, not searched experiment seeds.
- All EXP-004 model, optimizer, CutMix, drop-path, and SAM constants remain unchanged.

## Execution Environment

- Method: local single-process run from the repository root.
- Resources: physical GPU 0 only, NVIDIA H20 with approximately 97,871 MiB; eight existing DataLoader workers; no dependency changes.
- Estimated runtime: 300 seconds charged and approximately 440-500 seconds total; hard outer timeout 600 seconds.
- Log output: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log`, containing all stdout/stderr and serving as the source of truth.
- Full command: `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
- Monitoring: health patterns only. Intermediate test accuracy cannot trigger a decision.
- Tool skill: none; this is a local run monitored through its exec session and redirected log.

## Abort Criteria

- Abort before launch if GPU 0 is not the approximately 98 GB H20, branch/base is wrong, a protected file changed, any dependency changed, or a static/policy/math/CutMix/GPU/SAM smoke fails.
- During the run, stop on traceback, CUDA/OOM error, nonfinite loss or gradient, a hidden-mix assertion, CutMix-generator parity failure, mix/SAM overlap, or no progress long enough to make the 600-second timeout unavoidable.
- Exit 124, any nonzero exit, missing summary, charged time outside 299.5-301.0 seconds, total time at or above 600 seconds, a policy counter outside preregistered bounds, or parameter-count change is a protocol failure.
- Do not abort for weak intermediate accuracy. No metric-driven retry is allowed. A deterministic code or infrastructure defect before any valid summary may be repaired once and documented. An exact-count failure may only be repaired by fixing code, never by changing expected counts or seeds. After a valid summary the result is final.

## Verification Protocol

### Verification Procedure

1. **Parent and scope** (10-second timeout):
   - Resolve `PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-/root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2}"`, require executable `$PLUGIN_ROOT/skills/shared/scripts/tree.sh`, and query node 004; require stored `metric=95.40`, `commit=1a8d0de`, and extendable true. Derive the threshold as `Decimal("95.40") + Decimal("0.10") = Decimal("95.50")`.
   - Run `git branch --show-current`, `git rev-parse --short HEAD`, `git diff --name-only`, and `git status --short`; require EXP-006 at parent commit and exactly `train.py` as the tracked diff.

2. **Static checks** (60-second timeout):
   - Run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, and `git diff --check`; require exit 0. Do not run `ruff format`, because the inherited file is not format-clean.
   - Inspect `git diff -- train.py`; confirm only the approved policy, helper, forward API, counters, and config changed.

3. **Policy and RNG smoke** (120-second timeout):
   - Run an inline `uv run python` simulation of exactly 100,000 early batches with fresh fixed generators. Require exact counts: selected `49,769`, CutMix `37,141`, manifold `12,628`, boundary 2 `6,329`, and boundary 4 `6,299`; require lambda mean `0.500106625` and mean `min(lambda, 1-lambda)` `0.312155088` within fixed numerical tolerance. These values and seeds are immutable after preregistration.
   - Load the actual parent `train.py` from commit `1a8d0de` with `git show`, execute it under a non-main module namespace, and compare its CutMix helper against the refactored helper over a fixed mixed clean/selected prefix. Require equal decisions, specs, transformed outputs, and bitwise seed-42 CPU/CUDA generator states at every shared step.
   - Test the four-uniform Beta helper against hand-computed values, exactly four-draw advancement, finite strict `(0,1)` outputs, deterministic replay, fixed seeded sample moments including both lambda mean and mean `min(lambda, 1-lambda)`, and unchanged global CPU/CUDA RNG states.

4. **Forward and loss semantics** (120-second timeout):
   - In eval mode, require bitwise equality between `model(x)` and an explicit default-mix call. Invalid partial arguments, boundary, shape/device, or lambda must raise.
   - With source-coded toy activations and cyclic/fixed-point permutations, prove exactly one out-of-place interpolation after block 2 or 4, pristine paired sources, channels-last output layout, correct target orientation, exact two-term loss, and gradients to both source rows.
   - Re-run CutMix clipped-area, orientation, zero-area, aliasing, and deterministic-spec tests; its helper output must match the inherited implementation for fixed specs.

5. **GPU integration and parent invariants** (180-second timeout):
   - On `CUDA_VISIBLE_DEVICES=0`, run full-WRN BF16/channels-last clean, CutMix, block-2 manifold, block-4 manifold, and scheduled SAM steps. Require finite losses/gradients, one forward for every non-SAM step, no mix/SAM overlap, unchanged parameter count 2,748,890, and no global RNG consumption from private helpers.
   - Require clean/default forward parity; six drop-path draws for every training mode; ordinary/manifold BatchNorm buffers update once; and the unchanged SAM pulse still gives 0.05 perturbation, replayed CUDA RNG, one BN update, exact restore, and one optimizer update.
   - Run a short synchronized latency comparison and require manifold step median no more than 1.10x clean step median; this is a non-metric feasibility gate.

6. **Hardware and full run** (610-second timeout):
   - Run `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader`; require physical index 0 is NVIDIA H20 with about 97,871 MiB.
   - Ensure `run.log` is absent, then launch exactly once with `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`; require exit 0.

7. **Protocol integrity before metric** (30-second timeout):
   - Extract startup config, `mix_policy:`, `cutmix:`, `manifold:`, `sam:`, all eval lines, and the complete summary.
   - Require fixed constants/seeds; early selected ratio `[0.47,0.53]`; CutMix marginal ratio `[0.34,0.41]`; manifold marginal ratio `[0.10,0.15]`; each boundary marginal ratio `[0.045,0.080]`; manifold selected share `[0.21,0.29]`; discarded CutMix specs exactly equal manifold batches; mean lambda `[0.485,0.515]`; mean `min(lambda, 1-lambda)` `[0.300,0.325]`; zero mix after progress 0.75; SAM first progress `[0.7500,0.7520]`, even first step, applied/eligible ratio `[0.499,0.501]`; 299.5-301.0 charged seconds; total below 600; at least 24,000 steps; 2,748,890 parameters; every summary key exactly once; eval count exactly equals `num_epochs`.

8. **Primary verdict** (10-second timeout):
   - Run `grep '^best_test_acc:' run.log`; parse the percentage exactly once.
   - `>=95.50%` passes the necessary metric condition; `<95.50%` is no-improvement. Do not rerun, tune, change the share/boundaries/alpha, or choose another seed.

9. **Cleanup** (10-second timeout):
   - After recording all evidence in `03-execute.md` and `04-analysis.md`, run `rm -f run.log`; require no transient log or unintended tracked change remains.

### Informational Metrics (Optional)

- `final_test_acc`, `final_test_loss`: final summary; compare with best and parent loss 0.1654.
- `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`: final summary; compare with parent 300.0/457.3/1.2/1,190.5.
- `num_epochs`, evaluation count, `num_steps`, `num_params`: final summary and eval-line count; compare with parent 132/132/25,560/2,748,890.
- Policy exposure: final audit lines for early eligible/selected/clean, CutMix, manifold, boundaries, discarded specs, mean lambda, and mean `min(lambda, 1-lambda)`.
- SAM exposure: final audit line; compare transition and period-two arithmetic with EXP-004.
