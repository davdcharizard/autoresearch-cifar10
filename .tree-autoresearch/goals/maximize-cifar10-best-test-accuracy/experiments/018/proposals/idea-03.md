# Proposal: Direct Canonical Lookahead on EXP002

## Summary

Wrap EXP002's unchanged Nesterov SGD trajectory with canonical parameter-only Lookahead using fixed `k=5` and `alpha=0.5`. Keep one FP32 slow copy for every trainable parameter. After each fifth ordinary optimizer step, update `slow += 0.5 * (fast - slow)` and copy slow back into fast in place; retain the inner SGD momentum buffers exactly. BatchNorm running buffers are not slow state and remain live. Evaluation uses the slow parameters. Because the inherited epoch has 195 steps and `195 % 5 == 0`, every complete-epoch evaluation is already synchronized; only a budget-truncated final epoch can require a temporary slow-parameter swap and exact fast restore.

Only `train.py` changes. Architecture, initialization, CutMix/drop-path/data RNG, BF16/channels-last execution, learning-rate schedule, effective coupled weight decay, fixed seed 42, 300-second charged budget, and inherited evaluator/cadence/max-selection remain unchanged. All Lookahead interpolation work occurs inside the charged step. Every GPU command exposes only physical GPU 0.

## Rationale

Lookahead (`knowledge/papers/lookahead-optimizer.md`, NeurIPS 2019) reports CIFAR improvements and reduced optimizer variance using common `k=5`, `alpha=0.5` settings with no extra forward. EXP002's best-to-final gap is small but later SAM and EMA gains show that its optimizer trajectory still benefits from geometry and variance control. Direct use here isolates Lookahead from EXP011's separate evaluation EMA and avoids a nested smoother.

The mechanism uses abundant H20 memory and one sparse fused parameter interpolation every five updates. Unlike checkpoint EMA, the slow trajectory feeds back into subsequent optimization; unlike SAM, it adds no backward. The counter-case is that time-cosine SGD already becomes stable late, retaining fast momentum after a large parameter interpolation can create mismatch, and five-step averaging may reduce useful movement or duplicate the benefit of ordinary Nesterov.

## Fixed Mechanism

- Initialize slow parameters by cloning all 44 model parameters after model construction and before training, without consuming RNG.
- Use `LOOKAHEAD_K=5`, `LOOKAHEAD_ALPHA=0.5`; never tune them from test or preflight results.
- Run the parent `optimizer.step()` first on every batch. On steps divisible by five, apply one `torch._foreach_lerp_(slow, fast, 0.5)` followed by `torch._foreach_copy_(fast, slow)` under `torch.no_grad()`.
- Preserve SGD momentum buffers, parameter order, and effective weight decay. Do not reset, interpolate, or rescale momentum.
- Exclude BatchNorm running means/variances and integer buffers from slow state; they stay consistent with the live forward path.
- Apply Lookahead throughout early mixed and late clean phases. No activation gate, adaptive alpha, checkpoint selection, or test-driven synchronization is allowed.
- Evaluate slow parameters at every inherited evaluation. Complete epochs require no swap because 195 is divisible by five; for an unsynchronized final partial epoch, save fast parameters, copy slow to the model, run the unchanged evaluator once, and restore fast exactly.

## Audits

Record total inner steps, expected/actual sync count `floor(num_steps/5)`, steps since the last sync, early-CutMix/early-clean/late-clean sync counts, and total slow/fast parameter elements. At fixed sparse cadence, accumulate FP64 squared slow-fast distance immediately before interpolation and squared interpolation displacement; do not retain per-step CUDA tensors or synchronize inside training. At exit require exact sync-count and path reconciliation, positive finite displacement, zero nonfinite slow/fast/optimizer state, and unchanged 2,748,890 model parameters. Report final normalized slow-fast distance and final phase, but do not use them to alter execution or verdict.

## Accuracy-Blind Feasibility

Before metrics, deterministically prove that initialization is RNG-neutral, the first four steps equal parent SGD, step five matches an explicit Lookahead reference, momentum is retained, BN buffers are excluded, and evaluation-facing forward behavior is unchanged. Use real BF16/channels-last clean and CutMix batches with evaluator/test-loader guards and zero accuracy values.

Run one decisive GPU-0 preflight with a long production trace and five alternating-order paired rounds reflecting the 37.5% CutMix, 37.5% early-clean, and 25% late-clean workload. Diagnostic finiteness must reduce into fixed device scalars; allocation baselines occur only after optimizer/slow/audit state exists, and no device tensor is retained per step. Require parent drift at most 4%, ratio MAD/median at most 1.5%, median candidate/parent charged latency at most 1.01, every ratio at most 1.03, projected steps at least 27,500, epochs at least 141, total below 600 seconds, exact counters, and no persistent allocation growth. A malformed pre-vector harness gets at most one documented repair; a complete numeric failure is decisive.

## Metric Decision

After a passing preflight, launch exactly once with `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`. Require 299.5-301.0 charged seconds, total below 600, one evaluation per epoch, a complete inherited summary, exact Lookahead audit reconciliation, and only tracked `train.py` changed.

The falsifiable formal prediction is `best_test_acc >=95.33%`, +0.10 over EXP002's 95.23. A 95.33-95.52 result is a local tree improvement but noise-limited. A result at least 95.53 gives stronger single-seed context; 95.61 matches the global best and 95.71 clears it by the goal resolution. A valid result below 95.33 rejects this fixed direct Lookahead composition. Achieved step/epoch dose, final CE, final accuracy, and final-16 mean/range/premium are mandatory interpretation context but cannot override the frozen formal goal verdict.

## Risk and Effort

Scientific risk is medium-high because feedback averaging can lag the wall-clock cosine schedule and momentum-state mismatch is deliberate under canonical Lookahead. Throughput risk is low-medium: slow state is only one model copy and interpolation is sparse, but compact-model foreach launches still require measurement. Implementation risk is low-medium due exact cadence and state-ownership requirements. Estimated effort is medium.
