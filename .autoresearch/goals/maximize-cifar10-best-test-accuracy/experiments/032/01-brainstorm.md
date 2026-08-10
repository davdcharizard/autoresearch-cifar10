# Brainstorm EXP-032
**Created**: 2026-08-06

## Web Search & Literature Review

- **Scheduled Restart Momentum** (arXiv 2002.10583) reports that scheduled momentum restarts can improve deep-network convergence on CIFAR/ImageNet. It studies a NAG-like algorithm, so it supports only the broad stale-velocity mechanism, not this literal PyTorch SGD reset.
- **SGDR** (`knowledge/papers/sgdr.md`) shows scheduled regime restarts can improve CIFAR anytime performance. The local recipe already has one synchronized objective/LR boundary; resetting state there is more targeted than introducing another LR cycle.
- **Weight averaging** (`knowledge/papers/weight-averaging.md`) motivates trajectory smoothing, but EXP018's uniform tail SWA failed and any EMA variant must avoid both backward bias and fixed-budget overhead.
- **CutMix** (`knowledge/papers/cutmix.md`) supports regional target mixing, but local p=0.5 is the strongest proven gain and both stronger mixing and geometry substitutions hurt fit.

## Experimental History Review

- EXP010's 94.15% recipe remains protected. EXP030 shows more weak-tail LR lowers training loss while worsening NLL/top-1, favoring less inherited transition motion rather than more refinement amplitude.
- EXP031 invalidated hard-max residual pooling even after aggregate initialization scaling: sparse per-example ratios reached 4.34 and caused class concentration. Representation changes without intrinsic trajectory bounds remain high risk.
- Nesterov, Lookahead, and PNM failed as repeated high-LR optimizer paths. The deferred reset is distinct because it preserves ordinary SGD for the entire strong phase, deletes rather than amplifies state once, and first acts after the tenfold LR drop.
- Per-step additions face the EXP029 <=1% exposure gate. A one-time reset has negligible recurring cost; sparse EMA remains feasible only with fresh timing.

## Collected Ideas

- Reset every SGD momentum buffer exactly once at the accepted 80% data/target/LR transition, after switch evaluation and loader reconstruction but before the first weak update.
- Update a tail-only EMA shadow every 16 steps and evaluate it with online BN buffers at the unchanged evaluation opportunities, avoiding EXP018's uniform average but adding state/copy overhead.
- Change CutMix alpha from 1.0 to 0.5 at fixed p=0.5, altering box-area geometry without changing mixed frequency or GPU cost; this is cheap but its direction below the proven point is unsupported.

## Combinations

- Momentum reset plus sparse EMA could remove stale incoming velocity and smooth newly accumulated weak-tail noise, but combining two state mechanisms would destroy attribution. Test the reset alone first.
- Alpha-0.5 CutMix plus momentum reset could change both the learned representation and boundary state; the reset cannot diagnose whether geometry helped, so keep them separate.

## Candidate Ideas

### Reset SGD Momentum Once at the 80% Boundary
**Summary**: Preserve accepted ordinary momentum through all strong training, then zero all 59 live momentum buffers once after the switch evaluation/weak-loader rebuild and before the first LR-0.01 weak hard-label update. Full specification: `experiments/031/proposals/idea-01.md`.

**What it targets**: Stale high-LR composite-objective velocity crossing into a new view/target regime.

**Reasoning**: EXP030's aggressive tail fit harmed generalization, while the accepted quench scales but does not delete inherited velocity. Resetting is a bounded low-LR intervention, adds no recurring work, does not move parameters, and cannot affect strong-phase class geometry. Its main weakness is a short-lived, sign-ambiguous effect roughly comparable to one strong update.

**Sources**: Scheduled Restart Momentum; SGDR; EXP010, EXP020/022/028, EXP030; `experiments/031/proposals/idea-01.md`.

**Estimated Effort**: medium.

**Risk Assessment**: Medium; inherited velocity may be useful and the effect may be too small, but safety and attribution are unusually clean.

### Tail-Only Sparse EMA Every 16 Steps
**Summary**: From the 80% switch onward, update a shadow parameter EMA every 16 optimizer steps and evaluate the shadow at the existing opportunities using online BN buffers, with the online model retained for training.

**What it targets**: Late weak-tail variance and the 0.11-point regression seen in EXP030.

**Reasoning**: EMA weights recent iterates more than EXP018's harmful uniform SWA window and sparse updates may fit the 1% overhead budget. However EXP010 ends at its best, BN consistency is imperfect, and evaluator selection plus shadow-copy timing make this materially more complex than a one-time state reset.

**Sources**: `knowledge/papers/weight-averaging.md`; EXP010, EXP018, EXP029, EXP030.

**Estimated Effort**: high.

**Risk Assessment**: Medium-high; likely overhead and averaging bias, with no local evidence that the accepted tail needs smoothing.

### CutMix Alpha 0.5 at Fixed Probability
**Summary**: Change only `CUTMIX_ALPHA` from 1.0 to 0.5 while retaining p=0.5 and the full accepted curriculum, biasing sampled boxes toward smaller or larger areas.

**What it targets**: Regional mixing geometry and strong fit without changing mixed-batch frequency or GPU work.

**Reasoning**: It preserves the proven CutMix mechanism and is worker-side, but no local evidence establishes the direction; EXP026's Mixup geometry substitution and EXP031's localized-feature failure argue against another weakly supported geometry change.

**Sources**: `knowledge/papers/cutmix.md`; EXP010, EXP011, EXP026, EXP027, EXP031.

**Estimated Effort**: medium-high due exact-corpus policy and timing gates.

**Risk Assessment**: Medium-high; likely parameter chasing around a proven alpha-1 point.

## Review

Claude's independent review (`01-idea-review.md`) selected **Reset SGD Momentum Once at the 80% Boundary**, scoring evidence/reasoning 8/10 and impact 5/10. Sparse EMA scored 3/10 and 4/10 because EXP010's tail is monotone-improving, so averaging pulls backward as in EXP018, while online BN buffers mismatch shadow weights. CutMix alpha 0.5 scored 3/10 and 4/10 because no directional hypothesis supports moving from the proven alpha-1 point and nearby stronger/altered geometry has hurt.

The reset's limitation is explicitly adopted: its direct inherited-buffer contribution decays below 1% in about 44 weak steps and cumulative displacement is only roughly one strong update. A bare 94.25-94.35 pass is therefore protocol-valid but weak single-seed evidence. That ceiling does not undermine execution soundness: the intervention is exactly once, after the strong phase and tenfold LR drop, deletes state without relocating parameters, and cannot create the recurring high-LR class-geometry failures before the boundary. The full EXP031 proposal's copied-state, recurrence, concentration, spike, lifecycle, and no-rescue gates remain mandatory.

## Idea Evaluation

- **Reset SGD momentum once at 80%** — Advance. It is novel locally, zero-overhead, directly motivated by EXP030, and cleanly separated from every recurring optimizer failure.
- **Tail-only sparse EMA** — Reject. The accepted endpoint is already best; recent-weight averaging still biases backward and introduces BN/throughput complexity.
- **CutMix alpha 0.5** — Reject. This is unsupported tuning of the strongest proven augmentation knob and likely shifts toward more extreme regional targets.

## Chosen Idea
**Selected**: Reset SGD Momentum Once at the 80% Boundary

**Why this idea**:
It is the only candidate with a rigorous, isolated mechanism and no recurring cost: remove strong/composite-objective velocity exactly when the data, targets, and LR change, while leaving all accepted strong learning and subsequent SGD intact. Its likely effect is small, but the result will be causally clean and its safety can be established from one copied mature boundary state.

**Hypothesis**:
Zeroing all 59 SGD momentum buffers once after the 80% switch evaluation/weak-loader rebuild and before the first LR-0.01 weak update will reduce stale transition motion, preserve exposure, improve weak-tail generalization, and raise seed-42 `best_test_acc` from 94.15% to at least 94.25%. Point prediction is 94.27%; a valid miss rejects the full reset without partial/delayed tuning.
