# Brainstorm EXP-007
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **PyTorch `torch.compile` docs / TorchInductor (official)** (standard core-torch feature, torch 2.9.1):
  `torch.compile` traces the model and fuses pointwise ops (BN, ReLU, residual-add) into conv epilogues, cutting
  the number of dispatched CUDA kernels. The biggest wins are on **launch/overhead-bound** workloads (many small
  kernels, GPU underutilized) — exactly this project's regime at k=4. `mode="reduce-overhead"` additionally uses
  CUDA graphs to amortize launch cost. No new dependency (core torch).
- **Goyal et al. 2017, "Accurate, Large Minibatch SGD" (linear LR scaling rule)**:
  When increasing batch size by k, scale LR by k (with warmup) to preserve the optimization trajectory — used to
  raise GPU utilization (bigger batches) without losing accuracy.
- **Hu et al. 2018, SENet** (carried from brainstorm-006): channel recalibration, ~+0.3–0.5pp on CIFAR ResNets,
  but adds many small kernels per block.

## Experimental History Review

Current best **96.00%** (EXP-003, commit f59de56). Trajectory: recipe (000) → width k=4 (001) → Cutout (002) →
GPU-Cutout efficiency (003) → [k=6 ✗ 004] → [WD 5e-4 ✗ 005] → [EMA ✗ 006].

**Exhausted / saturated axes** (do NOT revisit):
- Width — k=4 is the sweet spot; k=6 compute-bound, underfits (goal-learnings § Failed/Patterns, EXP-004).
- Regularization — Cutout + label smoothing + WD near-saturated (EXP-005).
- Weight averaging — EMA redundant with cosine-to-0 (EXP-006); the annealed endpoint is already "averaged".

**Strongest causal lever in this project's own history**: *more epochs of the proven recipe*. EXP-003 went
54→77 epochs (by removing the Cutout CPU bottleneck) and gained **+0.58pp** → 96.00. goal-learnings § Patterns:
"under the 300s wall-clock, augmentation/kernel EFFICIENCY is itself an accuracy lever … keep per-step ops
vectorized/on-GPU; freed throughput re-enables capacity/aug."

**Key regime fact** (project-insights + EXP timing): at k=4 the H20 is **memory/launch-bound** — dt ~10–11ms/step
at batch 128 implies only ~45% compute utilization (a 4.3M-param 32×32 net's fwd+bwd is ~0.45 TFLOP/step; ~4.5ms
ideal vs 10ms measured). VRAM is free (507 MB / 98 GB). Noise floor ~0.2pp (target ≥~0.3pp expected gain).

**Untried gap**: nobody has attacked per-step *overhead* directly (kernel fusion / CUDA graphs) or raised batch
utilization — both convert the idle ~55% into more epochs of the identical 96% recipe.

## Candidate Ideas

### 1. `torch.compile` the training model (kernel fusion to cut launch overhead)
**Summary**: Wrap the training model with `torch.compile` (TorchInductor) so per-step kernel launches are fused
and reduced, lowering dt and fitting more epochs in the 300s budget. Compile the training model handle
(`compiled = torch.compile(model)`); keep the **original eager `model` for eval** (compile shares parameters via
`._orig_mod`, so eval sees trained weights without triggering eval-time recompiles on the variable last test
batch). `drop_last=True` already fixes the training batch shape → no train-time recompiles. Everything else in
the recipe is **byte-for-byte unchanged** (k=4, Cutout, PEAK_LR 0.2, WD 1e-4, label smoothing, batch 128, bf16,
channels_last, Nesterov, cosine, seed 42). Start with default mode (robust pointwise fusion); `reduce-overhead`
(CUDA graphs) is a more aggressive follow-up if default underwhelms.

**Reasoning**: This is the cleanest possible instance of the project's strongest established lever — it adds
*more epochs of the exact 96.00% recipe* by cutting idle launch overhead, without touching optimization dynamics
(so attribution is unambiguous, unlike batch/LR changes). The regime is launch-bound (~45% util), which is
precisely where `torch.compile`/fusion delivers the largest speedups. EXP-003 already proved more epochs → higher
acc on this regularized, not-fully-converged model (final_test_loss 0.204 ⇒ room to train more). No new dep.

**Sources**: PyTorch `torch.compile` docs; goal-learnings § Patterns (efficiency = accuracy lever, EXP-003);
project-insights (launch-bound, VRAM free); experiment-index EXP-003 row.

**Estimated Effort**: low (2–3 lines: compile the model, eval the original handle).

**Risk Assessment**: (a) Compile *cost is charged to the 300s budget* (first step(s) include trace/compile,
~10–40s) — caught honestly by the budget; the speedup must repay it (at ~30s compile and a ~30–40% step speedup
over the remaining ~270s, it repays many-fold). (b) Compile could fail or graph-break (autocast/channels_last
interactions) — mitigated by a Milestone-1 sanity check that compiles + runs one fwd/bwd before the full run, so
failures are caught cheaply. (c) If the model is *not* actually launch-bound, the gain is small → no-improvement
(graceful). Worst case: a crash caught at sanity check (revise/fall back) or a graceful no-improvement.

### 2. Larger batch (256) + linear LR scaling (0.2→0.4)
**Summary**: Double batch to 256 and scale PEAK_LR to 0.4 (Goyal linear rule) with the existing 5% warmup. Bigger
batches amortize per-launch overhead and fill the GPU → higher img/s → more epochs.

**Reasoning**: Another route to higher utilization → more effective training (the EXP-003 lever). VRAM is free
(~1 GB at batch 256). The linear-scaling rule + warmup is designed to preserve the trajectory.

**Sources**: Goyal 2017 (linear scaling); EXP-003 (epochs→acc); project-insights (launch-bound, VRAM free).

**Estimated Effort**: low (two constants: BATCH_SIZE, PEAK_LR).

**Risk Assessment**: Two coupled changes (batch + LR) → weaker attribution than idea 1. LR 0.4 is aggressive —
divergence risk (→ crash) despite warmup/BN. Large-batch training has a known small generalization gap per update
that could offset the throughput gain → net-neutral. Less clean than compile, which leaves dynamics untouched.

### 3. Squeeze-and-Excitation blocks on k=4
**Summary**: Add an SE module (GAP → FC→ReLU→FC→sigmoid → channel scale, r=16) to each BasicBlock. Carried from
brainstorm-006.

**Reasoning**: A fresh architectural-efficiency axis; ~+0.3–0.5pp nominal on CIFAR ResNets at fixed width.

**Sources**: Hu et al. 2018; brainstorm-006 § Candidate 2.

**Risk Assessment**: **Directly opposes the launch-bound diagnosis** — SE adds ~5 small kernels × 18 blocks per
forward, raising launch pressure and likely *cutting* epochs (the EXP-002/EXP-004 epoch-loss trap). Its nominal
gain is conditional on epochs holding, which is exactly what's at risk here. Worse fit than ideas 1–2 this loop.

## Idea Evaluation

All three respect hard constraints (train.py-only, no new deps, single H20/300s, eval once/epoch, seed 42, no
reward hacking — they speed up or restructure *training*, not the frozen eval). Decision criteria: expected gain
above the ~0.2pp noise floor, attribution cleanliness, and fit to the launch-bound diagnosis.

- **Mechanism / fit to bottleneck**: the metric is bottlenecked by *effective training within 300s*, and the
  hardware is *launch-bound*. Idea 1 (`torch.compile`) attacks launch overhead head-on while leaving the
  optimization recipe identical — the tightest possible match between diagnosis and intervention. Idea 2 raises
  utilization too but perturbs the optimizer (LR/batch). Idea 3 *increases* launch count — wrong direction.
- **Evidence**: idea 1 is backed by both general knowledge (compile wins are largest when launch-bound) and this
  project's own strongest result (EXP-003: more epochs → +0.58pp). Idea 2 shares the epochs→acc evidence but adds
  LR-stability uncertainty. Idea 3's evidence (SENet) is from settings without this budget/launch constraint.
- **Attribution**: idea 1 changes nothing in the math (same weights trajectory, just computed faster) → a clean
  single-variable test. Idea 2 is two coupled knobs. Idea 1 wins.
- **Risk profile**: idea 1's failure modes are either caught cheaply (sanity-check compile failure) or graceful
  (small speedup → no-improvement). Idea 2 risks divergence (crash). Idea 3 risks self-defeating epoch loss.

`torch.compile` is the highest risk-adjusted EV: best diagnosis fit, cleanest attribution, strongest in-project
evidence, cheap, no new dep, failures caught early. Batch+LR is the natural fallback if compile under-delivers;
SE is deferred (its throughput risk is worst exactly in this regime).

## Chosen Idea
**Selected**: `torch.compile` the training model (kernel fusion to cut launch overhead)

**Why this idea**: It directly attacks the established bottleneck (launch-bound ~45% utilization under a 300s
wall-clock) using the project's strongest proven lever (more epochs of the proven recipe, EXP-003) while leaving
the optimization recipe byte-for-byte unchanged — the cleanest possible attribution. It is low-effort, uses only
core torch (no new dependency), composes with the existing bf16 + channels_last, and its failure modes are either
caught cheaply at a sanity check or degrade gracefully to no-improvement.

**Hypothesis**: Compiling the training model will lower per-step time enough (via fused/fewer kernels) to fit
materially more epochs in 300s (target ≳90 vs EXP-003's ~77) even after paying the one-time compile cost, and
those extra epochs of the identical 96.00% recipe will lift `best_test_acc` past the +0.1pp bar (≥96.10%,
expected ~96.2–96.6%) at unchanged VRAM headroom and unchanged num_params (4,299,866).
