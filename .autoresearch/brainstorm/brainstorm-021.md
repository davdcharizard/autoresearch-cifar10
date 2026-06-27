# Brainstorm EXP-021
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

No new external searches — implementation-level ideation grounded in PyTorch's own documented compile/optimizer machinery plus the knowledge base:

- **torch.compile modes** (PyTorch docs, pytorch.org/docs/stable/generated/torch.compile.html): `mode="max-autotune"` enables Triton template autotuning for matmuls/convs and CUDA graphs; `mode="reduce-overhead"` enables CUDA graphs to eliminate per-kernel launch overhead. EXP-006 (our +0.48pp compile win) used the DEFAULT mode — the autotuning/graphs tiers above it have never been probed. CUDA graphs fit our loop: static shapes every step (batch 512, drop_last=True), optimizer outside the compiled region, weights updated in-place at static addresses, eval on the eager `base_model` reference is untouched.
- **Fused SGD** (PyTorch docs, torch.optim.SGD `fused=True`): single fused CUDA kernel for the SGD update (supports momentum/nesterov/weight_decay) vs the default multi-kernel foreach path; `optimizer.step()` runs INSIDE our timed dt. Mathematically identical update rule — pure throughput.
- **cifar10-fast / airbench** (knowledge/README.md References): both speedrun lineages treat step-time engineering as the dominant lever for tiny models — consistent with our EXP-006 result that throughput at fixed hyperparameters converts to accuracy.

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (EXP-006). **Fifteen consecutive misses (EXP-007…020).**
- **Certified local optimum** (goal-learnings § Patterns High): every constant bracketed both directions; four structural perturbations below baseline (EXP-017 free params, EXP-018 zero-γ, EXP-019 whitening, EXP-020 reference-faithful projection shortcuts).
- **Deferral law** (project-insights High, 8 confirmations / 5 mechanism classes): any change priced in early heat or epochs loses; a change must be free in BOTH.
- **The one validated +pp mechanism**: throughput at byte-identical hyperparameters (EXP-006: +25 epochs → +0.48pp). EXP-012's caveat: throughput that forces hyperparameter changes is metric-neutral.
- **CORRECTION to exp-report-020 § Next Steps #1 (GPU-side augmentation — DISCARDED on a false premise)**: re-reading `train.py` L215–238, `t0 = time.time()` is set AFTER `for inputs, targets in train_loader:` yields — loader stalls are entirely OUTSIDE dt and outside the 300s timed budget; they consume only the 600s wall cap. EXP-013's infra-errors entry confirms empirically: stalls grew 50→197s while epochs stayed 139. Therefore eliminating loader stalls adds ZERO epochs — the "~50s ≈ +20 epochs" premise was wrong. GPU-side augmentation would also move augmentation cost INTO dt (fewer epochs) if applied after t0. Idea discarded, not deferred; recorded here so no future loop re-walks it.
- **What IS inside dt** (train.py L216–237): H2D copy + channels_last conversion (~0.5ms), LR set, zero_grad, compiled forward+backward (~18–19ms), `optimizer.step()` (~1–2ms, foreach), synchronize. The untried dt levers: compile tier above default (max-autotune / CUDA graphs) and fused SGD. Both leave every hyperparameter, the augmentation distribution, and the update rule byte-identical.
- **Untried gaps remaining**: (a) dt engineering via compile mode + fused optimizer; (b) heat-constant momentum+peak trade — the only never-touched constant, admissible only as a compensated trade.

## Candidate Ideas

### 1. Step-time engineering: torch.compile(mode="max-autotune") + SGD(fused=True)
**Summary**: Two pure-throughput changes to the same timed region, zero hyperparameter changes: (a) compile the training step with `mode="max-autotune"` (Triton autotuned conv/matmul templates + CUDA graphs) instead of default; (b) construct the SGD with `fused=True` (single fused update kernel inside dt) instead of the default foreach path. The math is identical — same update rule, same augmentation, same schedule; only ms/step changes. Gate at step ~100: if measured windowed dt is not ≤ ~21.5ms (≥4% faster), the gain cannot clear the bar — kill early and save wall clock.

**Reasoning**: This is the only mechanism with a validated positive sign on this goal (EXP-006: throughput at fixed hparams → +0.48pp from +25 epochs; EXP-012 only failed because LR had to change — here nothing changes). It is free in BOTH early heat and epochs (it ADDS epochs) — the only intervention class that passes the EXP-020-sharpened transfer rule. dt 22.4→20.5ms would give ~147 epochs (+10); EXP-006 arithmetic prices that at ≈ +0.2pp, above the +0.1 bar. Both sub-changes target real dt components: max-autotune attacks the ~18–19ms compiled compute (Triton conv templates can beat cuDNN at small channel counts on Hopper-class parts) and CUDA graphs eliminate launch overhead across the ~10² kernels/step; fused SGD collapses the multi-kernel optimizer pass over 65 param tensors.

**Sources**: PyTorch torch.compile / SGD docs (above); goal-learnings § Patterns High (EXP-006 conversion, EXP-012 caveat); project-insights Medium (dt must be spot-measured in the target regime — gate built in); train.py L216–237.

**Estimated Effort**: low — two-argument diff (`mode=` in torch.compile, `fused=True` in SGD) plus the dt gate during monitoring. Risk lives in compile time, not code.

**Risk Assessment**: (1) max-autotune compile/autotune time is the main hazard — could add 30–120s to startup; budget check: baseline total 495s leaves ~105s headroom under the 600s cap, and the warm inductor cache halves repeat cost. Mitigation: if startup exceeds ~120s or total projects >600s, fall back to `mode="reduce-overhead"` (cheap compile, CUDA graphs only). (2) CUDA-graphs constraints (static shapes, in-place weight updates) are all satisfied by our loop; the eager-eval `base_model` shares weights at static addresses — unaffected. (3) Honest probability that max-autotune+fused beats the already-tuned default+cudnn-benchmark by ≥4%: maybe 35–45% — the gate converts a miss into a cheap early kill rather than a wasted run. (4) Failure mode graceful: same recipe, fewer-or-equal epochs, converged no-improvement.

### 2. CUDA graphs only: torch.compile(mode="reduce-overhead")
**Summary**: The conservative tier of idea 1 — keep default kernel selection (cuDNN benchmark already autotunes convs) and add only CUDA graphs to erase per-kernel launch overhead; compile time stays ~EXP-006-like (~23s cold).

**Reasoning**: If the model is launch-overhead-bound at all (plausible at 22ms/step with ~10² kernels), graphs capture that gain without max-autotune's compile-time risk. Smaller expected dt gain (~2–5%) but near-zero startup risk.

**Sources**: PyTorch docs (above); EXP-006 startup measurements (22.8s cold / 13s warm).

**Estimated Effort**: low — one argument.

**Risk Assessment**: Expected gain likely sits at-or-below the 4% epochs-to-bar threshold — high chance of a converged within-noise miss. Strictly dominated by idea 1, which includes graphs AND autotuning with a built-in fallback to this exact configuration.

### 3. Heat-constant momentum trade: MOMENTUM 0.9→0.95 with PEAK_LR 0.4→0.2
**Summary**: Hold effective per-step size lr/(1−β) at the certified 4.0 while doubling the gradient-averaging horizon; tests whether smoother update directions at identical integrated heat lengthen/raise the converged plateau the max-statistic harvests.

**Reasoning**: Momentum is the only recipe constant never touched; single-knob momentum moves are priced by the closed heat axis, so only the compensated trade is admissible.

**Sources**: goal-learnings § Failed Approaches Medium (heat axis closed twice); exp-report-020.md § Next Steps #2.

**Estimated Effort**: low — two constants.

**Risk Assessment**: No comparable-regime evidence; the lr/(1−β) equivalence is first-order only, so the probe risks re-measuring the heat optimum with extra trajectory variance. Graceful failure but lowest prior of the three.

## Idea Evaluation

**Evidence strength**: Idea 1 rests on the strongest evidence available anywhere in the remaining space: an IN-PROJECT validated mechanism (EXP-006's throughput→accuracy conversion at fixed hyperparameters) rather than external fixed-epoch literature — the evidence class that has now failed to transfer four times (RegNet/EXP-017, Bag of Tricks/EXP-018, airbench/EXP-019, WRN/EXP-020). Idea 2 shares the mechanism but caps its own upside. Idea 3 has no direct evidence.

**Mechanism clarity**: Idea 1: smaller dt → more steps in the fixed 300s of accumulated dt → more epochs at byte-identical hyperparameters → EXP-006 conversion. Sharp, quantified (4% dt ≈ bar), and gate-measurable at step 100. Idea 3's mechanism is speculative.

**Expected impact**: Idea 1: +10–15 epochs if max-autotune lands ≈ +0.2–0.3pp; Idea 2: likely within noise; Idea 3: likely within noise.

**Risk profile**: Idea 1's hazards are operational (compile time vs the 600s cap) with a defined fallback (reduce-overhead) and an early-kill gate — the failure mode is cheap and informative either way. No learning-dynamics risk at all: the trajectory is the baseline trajectory, just denser in wall time.

**Feasibility**: Two arguments + a gate. The synthesis-check discipline (15 misses) favors exactly this: the last validated mechanism, probed at its last untried tier.

Idea 2 folds into Idea 1 as its fallback configuration. Idea 3 is held as the next probe if dt engineering exhausts.

## Chosen Idea
**Selected**: Step-time engineering: torch.compile(mode="max-autotune") + SGD(fused=True)

**Why this idea**:
It is the only remaining candidate built on an in-project validated positive mechanism (EXP-006), the only class that is free in both early heat and epochs (it adds epochs), and it carries a built-in measured-dt gate that converts the likely failure mode (insufficient speedup) into a cheap early kill instead of a burned loop. Hyperparameters, augmentation, and update math stay byte-identical, so attribution is exact and there is zero deferral risk.

**Hypothesis**:
max-autotune kernel selection + CUDA graphs + fused SGD reduce per-step time from 22.4ms to ≤21.0ms (≥6%), fitting ≥147 epochs of the unchanged recipe into the 300s timed budget; by the EXP-006 conversion (epochs at fixed hyperparameters convert at roughly +0.02pp/epoch), best_test_acc ≥ 96.81 (baseline 96.71 + 0.1) with an unchanged converged-plateau shape. Falsifier at step ~100: windowed dt > 21.5ms → kill (cannot clear the bar); secondary falsifier: startup so long that total projects past the 600s cap → fall back to reduce-overhead.
