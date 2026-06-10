# Brainstorm EXP-036
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

<!-- Ideation only. Metric/direction/constraints/verification live in the goal file;
     baseline (96.22, commit 6c417a4) lives in experiment-indices/improve-cifar10-test-accuracy.tsv. -->

## Web Search & Literature Review

No new external search this loop — grounding from established literature I can cite directly + the project knowledge base (`knowledge/README.md`):
- **SAM — Sharpness-Aware Minimization (Foret et al., ICLR 2021)**: minimize the max loss in an ε-ball around the weights (`w ← w − ∇L(w + ρ·∇L/‖∇L‖)`), seeking *flat* minima that generalize better. Reliable **+0.3–1.0pp top-1** on CIFAR-10/100 at FIXED architecture — a genuine generalization gain, not a loss/calibration polish. Cost: **2 forward-backward passes per step** (~2× per-step compute). Standard CIFAR ρ ≈ 0.05.
- **LookSAM / periodic SAM (Liu et al., CVPR 2022; "Towards Efficient and Scalable SAM")**: applying the SAM ascent only every *k* steps (plain SGD otherwise) retains most of SAM's generalization benefit at a fraction of the compute — directly motivates a sparse-SAM design to dodge a compute wall.
- **Project knowledge**: `knowledge/papers/swa.md` (weight-averaging flat-minima lever — CLOSED here, loss-not-top1); `knowledge/papers/gradient-centralization.md` (optimizer-axis polish — CLOSED for top-1). SAM differs from both: it changes the *objective* (worst-case-in-ε-ball), not just gradient conditioning or trajectory averaging.

## Experimental History Review

**Current best / baseline**: 96.22% (EXP-012, commit 6c417a4); bar = 96.32 (+0.1). 35 experiments; the plateau is firmly established.

**Diagnosed binding constraint**: the net is **generalization-bound at fixed k=4 capacity** and **convergence-saturated** (~77 ep converges, baseline 91 ep — EXP-007). Top-1 gains require a *generalization* mechanism, not optimization polish or more capacity.

**The two hard walls bounding every idea (project-insights High Importance)**:
1. **Compute/epoch wall** (EXP-004/009/015/024): ANY non-trivial FLOP add → fewer epochs → under-train → regress, *regardless of the change's merit* (merit gets masked). Only pursue compute-NEUTRAL changes while launch-bound. Quantitatively: converges ~77 ep, baseline 91 ep → **~15% epoch slack** before under-training bites.
2. **Polish-vs-top1 pattern** (EXP-006/019/020/023/026/030/031): compute-neutral *optimization* polish (EMA, SWA, GC, LS-down, Bag-of-Tricks) lowers loss/ECE but NOT top-1. Spans weight-averaging + init/WD + optimizer/gradient-dynamics axes.

**CLOSED axes (do not revisit)**: capacity (k>4 compute-wall); ALL augmentation (TA+Cutout=ceiling; policy/Mixup/CutMix/Cutout-size/cooldown-schedule incl. EXP-035 reheat); ENTIRE LR-schedule (peak/floor/shape/SGDR); all regularizer-adds (WD/dropout/LS — convergence-bound, adds underfit); architecture (SE/SiLU/preact/ResNet-D/BlurPool/multi-scale-head/large-batch); weight-averaging & gradient-centralization (polish cluster).

**The untouched lever**: the **optimizer OBJECTIVE** — no experiment has changed *what* is minimized (only how gradients are conditioned, via GC). SAM (flat-minima objective) is the single most-cited generalization lever in the literature and has never been tested here. Its blocker is the compute wall (2× cost) — which the ~15% epoch slack lets a *sparse* variant sidestep.

## Candidate Ideas

### 1. Periodic / sparse SAM (Sharpness-Aware Minimization on a fraction of steps)
**Summary**: Add SAM to the SGD update, but apply the (expensive) sharpness ascent only every *N*-th step; plain Nesterov-SGD on the other steps. On a SAM step: (a) first forward-backward → grad g; (b) perturb weights `w += ρ·g/(‖g‖+1e-12)` (global L2 norm over all params); (c) second forward-backward at the perturbed point → grad g_sam; (d) restore `w −= ρ·g/‖g‖`; (e) `optimizer.step()` using g_sam. Keep everything else fixed (k=4, TA+Cutout, cosine-to-0, torch.compile, bf16, seed 42). Choose **N=5** (SAM on ~20% of steps): mean per-step cost ≈ (1−f)+2f = 1.2× → ~76 epochs ≈ the ~77-ep convergence point, so the test is near-throughput-fair (respects the compute-wall insight quantitatively). ρ=0.05 (standard CIFAR SAM radius).

**Reasoning**: The model is generalization-bound at fixed capacity — and SAM is the canonical fixed-architecture generalization lever (flat minima → +0.3–1.0pp top-1 on CIFAR in the literature). It is mechanistically DISTINCT from every closed axis: it changes the minimized objective (worst-case loss in an ε-ball), not gradient conditioning (GC, closed), trajectory averaging (SWA/EMA, closed), the LR schedule (closed), or regularizer strength (closed). The polish-vs-top1 pattern does NOT obviously apply — SAM's reported effect is specifically on top-1 generalization, not just loss/ECE. Sparse application (LookSAM-grounded) is the key design move: it keeps epochs ≥ the convergence point so any result is a fair-ish test of the mechanism rather than a compute-confounded underfit.

**Sources**: SAM (Foret 2021); LookSAM/periodic-SAM (Liu 2022); project-insights High (compute-wall slack ~15%, EXP-007 convergence); generalization-bound diagnosis (goal-learnings).

**Estimated Effort**: medium — ~15–20 lines in the training loop (two-pass SAM step gated by `step % N`), pure torch (no new dep). Main risk is the SAM perturbation interacting with `torch.compile(reduce-overhead)` CUDA-graph capture across two forward calls per step.

**Risk Assessment**: (a) **Compute-wall confound** — if realized dt pushes epochs well below ~75, a regression is under-training, not a clean SAM null (mitigated by N=5≈1.2×; will verify epoch count and tune N up if dt is worse than predicted). (b) **torch.compile + in-place param perturbation** may break CUDA-graph replay or silently misbehave; mitigation: smoke-test the compiled two-pass step first, and if it fails, fall back to compiling only the forward / eager SAM (accepting some dt cost) or raise N. (c) SAM's literature gains use longer schedules; the shallow-net/short-budget transfer is uncertain (cf. the "ImageNet tricks don't transfer" Medium insight). Worst case: compute-confounded regression → informative (closes the flat-minima/SAM axis at this budget).

### 2. Full per-step SAM (every step)
**Summary**: The same SAM step applied on EVERY step (ρ=0.05). Cleanest test of the SAM mechanism.

**Reasoning**: Maximal flat-minima signal; the textbook SAM.

**Sources**: SAM (Foret 2021).

**Estimated Effort**: medium (same code, no `step % N` gate).

**Risk Assessment**: **High compute-wall risk** — 2× per-step cost → ~45 epochs (cf. EXP-009 compiled-k5 41 ep → 94.21 under-fit). The High-Importance compute-wall insight predicts this regresses via under-training regardless of SAM's merit, exactly like k=5/k=6/BlurPool. The result would be compute-confounded and uninformative about SAM itself. Dominated by candidate 1.

### 3. Proper per-channel input std-normalization (compute-neutral)
**Summary**: The code normalizes inputs by mean only (`std=(1,1,1)`, train.py L154). Switch to the standard CIFAR-10 per-channel std `(0.2470, 0.2435, 0.2616)` so inputs are unit-variance per channel. Zero compute cost (just transform constants).

**Reasoning**: Compute-neutral and explicitly flagged by project-insights as an untried convergence-neutral lever ("input normalization"). Unit-variance inputs better match the Kaiming-init assumption at conv1 and equalize the R/G/B channel scales the net currently sees unequally.

**Sources**: project-insights Medium (convergence-neutral levers list); standard CIFAR normalization.

**Estimated Effort**: low — one-line constant change.

**Risk Assessment**: Very likely **within-noise null** — conv1 is immediately followed by BN, which absorbs input scale, so the effect is mostly on conv1's effective LR and cross-channel balance. Safe (no crash/compute risk) but low expected impact (<0.1pp). Good as a cheap fallback, weak as a bar-clearer.

## Idea Evaluation

The binding constraint is generalization at fixed capacity, and the two walls (compute, polish-pattern) have closed every incremental lever. The question is which candidate best attacks generalization *without* tripping a wall.

- **Mechanism / expected impact**: Candidate 1 (sparse SAM) is the only one targeting generalization through a genuinely untouched mechanism (the minimized objective) with strong literature support for a top-1 (not loss) gain. Candidate 3 is compute-neutral and insight-endorsed but its mechanism is mostly absorbed by BN → expected sub-noise. Candidate 2 has the cleanest mechanism but is dominated by the compute wall.
- **Respecting the compute wall (decisive)**: Candidate 2 fails it (→~45 ep underfit). Candidate 1 is explicitly engineered around it using the project's own ~15%-epoch-slack data (N=5 → ~1.2× → ~76 ep ≈ converged). Candidate 3 is compute-free.
- **Risk profile**: Candidate 3 is safest (no crash, no compute risk) but lowest upside. Candidate 1 carries real implementation risk (SAM × torch.compile) and compute-confound risk, but both are mitigable (smoke-test; tune N; verify epochs) and the failure mode is an informative axis-closing null.
- **Information value**: Candidate 1 tests THE most direct lever for the diagnosed binding constraint — a result either way (gain → plateau broken; fair null → flat-minima/SAM axis closed at this budget) is high-value and non-redundant. Candidate 3's null is minor ("BN absorbs input scale"). Candidate 2's null is uninformative (compute-confounded).

Candidate 1 wins on mechanism alignment, expected impact, information value, and is the "think harder / radical" move the loop calls for after ~27 closed axes — while still honoring the compute-wall constraint by design.

## Chosen Idea
**Selected**: Periodic / sparse SAM (Sharpness-Aware Minimization on ~20% of steps, N=5, ρ=0.05)

**Why this idea**:
After ~27 mapped axes, the model is generalization-bound at fixed capacity and every compute-neutral *optimization-polish* and *regularizer* lever is closed. SAM is the single most-cited fixed-architecture *generalization* lever and is mechanistically untouched here (it changes the minimized objective, not gradient conditioning/averaging/schedule). Its only blocker — the 2× compute wall that doomed naive capacity adds — is sidestepped by the sparse design, sized directly from the project's own ~15% epoch slack (N=5 → ~1.2× → ~76 ep ≈ the ~77-ep convergence point). This makes it both the highest-upside and a fair-as-possible test, with an informative result in either direction.

**Hypothesis**:
Applying SAM (ρ=0.05) on every 5th step will steer SGD toward a flatter minimum that generalizes better, lifting best_test_acc above the 96.22 plateau — targeting ≥96.32 — while staying near-throughput-neutral (~75–78 epochs, params 4,299,866 unchanged). Falsified if (a) best_test_acc ≤ ~96.3 at ≥75 epochs (flat-minima/SAM axis then closed for this budget), or (b) realized epochs fall well below ~75 (compute-confounded — would require raising N or dropping to a cheaper SAM to retest). A regression with settled-but-not-higher behavior would distinguish "SAM doesn't help here" from "under-trained."
