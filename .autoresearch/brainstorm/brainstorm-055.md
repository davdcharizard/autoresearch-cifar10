# Brainstorm EXP-055
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external fetches; sources are the project record + model knowledge of a directly-relevant published method:

- **FreezeOut (Brock et al. 2017, arXiv:1706.04983 — model knowledge, CIFAR-validated)**: progressively freeze early layers during training, each layer following its own cosine anneal that completes at its freeze time (per-layer LR scaled so integrated heat is preserved), then excluded from backward. Published result: up to ~20% training-time reduction on CIFAR DenseNets/ResNets with ~no accuracy loss under FIXED-EPOCH budgets. Under THIS project's fixed-TIME budget, the same compute saving converts to extra steps instead of saved wall-clock — the conversion the fixed-epoch paper could not exploit.
- **ResNet stage FLOPs arithmetic (train.py)**: the three stages are FLOPs-balanced by design (64²·32², 128²·16², 256²·8² are equal), so stem+layer1 ≈ ⅓ of conv compute; freezing them removes their entire backward subgraph (both weight-grads and input-grads, since nothing below needs gradients) ≈ ~22% of step time ≈ ~4–5ms/step.
- **Project laws this must pass**: deferral (payoff arrives in-run: extra steps start at the freeze point); numerics (remaining layers' kernels unchanged); noise (per-step batch noise unchanged); heat (per-layer integrated heat preserved by the FreezeOut compensation — the global-heat bracket measured constants, never per-layer allocation); throughput-exhausted law (EXP-048 bounded NON-KERNEL overhead at 0.7% — freezing reduces the KERNEL WORK itself, a lever that law does not bound).
- **torch.compile caveat (in-project knowledge, infra-errors EXP-021 adjacency)**: flipping `requires_grad` mid-run invalidates dynamo guards → a CHARGED mid-run recompile (~tens of seconds) unless both graph variants are pre-compiled in the uncharged warmup region (the established EXP-006 warmup pattern extended to two variants: random data, no optimizer.step — same category of work).

## Experimental History Review

State after 55 indexed experiments: baseline 96.71 @ 1990397, bar ≥ 96.81, 48 consecutive non-improvements. Frontier after EXP-054:

- **Every catalogued axis is measured-closed**: recipe constants; loss axis both directions (050/051); structural classes (017–020/026/030/034/037/040–047); per-step throughput at the 99.3% kernel floor (048) with numerics closed BOTH directions (021 coarser −0.20, 054 finer null); gradient-noise bracket; averaging both kinds; augmentation dose-response; data order/coverage; BN/eval constants; compound-of-frees (053).
- **What no experiment has ever touched**: the ALLOCATION of compute across layers and time. Every run so far backpropagates through all 4.29M params on every one of ~13,400 steps. Freezing converged early layers in the anneal's tail is a compute-reallocation mechanism — it buys extra steps for the layers still learning, paid for by layers whose anneal has already completed. No closed law prices it: it is not aug-suppliable, not a noise change, not a numerics change, not a recipe constant, not a structural edit (the function class at eval time is identical).
- Relevant priors to respect: EXP-025/033 (tail-lightening of the DATA loses — but this construction keeps the data distribution at full pressure to the last step; what it lightens is the parameter set still moving); EXP-018 (init-time freezing = deferral — this freezes at the END, the opposite end of the schedule); max-statistic law (extra tail steps extend/raise the converged plateau, exactly what best-over-evals rewards).
- Protocol carry-overs: composite gates, trajectory criterion (048), replicate escalation for near-bar reads (052), step ledger with replication caveat (053).

## Candidate Ideas

### 1. FreezeOut-style tail freezing of stem+stage1 with per-layer heat compensation (time-keyed, dual-graph pre-compile)
**Summary**: Split params into group A (conv1, bn1, layer1 — ⅓ of conv FLOPs) and group B (rest). Group A follows a compressed schedule `lr_A(p) = lr_at(min(p / FREEZE_FRAC, 1))` (its full warmup+cosine completes by FREEZE_FRAC = 0.70 of the budget — integrated heat preserved per FreezeOut); at p ≥ 0.70 group A is frozen (`requires_grad_(False)`, BN modules of A kept in train mode so running stats keep tracking) and the step's backward drops A's entire subgraph (~22% of step time). Saved compute → ~+45% more steps for group B in the final 30% of the schedule (~+1,300 tail steps ≈ +13 tail epochs). Compile warmup extended to warm BOTH graph variants (frozen and unfrozen) on random data with no optimizer.step, so the p=0.70 flip hits a cached graph instead of a charged recompile.

**Reasoning**: This is the first candidate in ~20 loops with a mechanism NO standing law prices: compute reallocation across layers and time. Why it can pay: (a) the max-statistic rewards a longer, denser converged plateau — +13 tail epochs of full-pressure refinement for layers 2–3 plus +13 extra eval draws on the plateau; (b) per-layer heat is preserved (FreezeOut's own control), so group A converges on schedule rather than being abandoned; (c) external anchor at fixed epochs shows the freeze itself costs ~nothing — our regime ADDS the converted steps the paper threw away. Why it can fail: layer-1 tail refinement may be load-bearing in a way the data-side tail laws hint at (then the read is low — terminal closure of the class); or the compile dual-graph trick leaks a charged recompile (visible as a one-off mid-run stall + collapsed step ledger — integrity-classified, fix-or-fail). Honest expected effect: +0.05–0.15 if the freeze is free (13 tail epochs are worth less than 13 full-run epochs; bounded by tail-conversion uncertainty), bar-clearing only at the optimistic end — but every branch closes a never-touched class.

**Sources**: model knowledge: FreezeOut (Brock et al. 2017, arXiv:1706.04983); train.py stage arithmetic; goal-learnings EXP-025/033 (tail-pressure law — distinguished: data vs parameter lightening), EXP-018 (freezing at init = deferral; this is the mirror end), EXP-048 (kernel-floor law bounds overhead, not work); infra-errors EXP-021 entry (compile-guard recompile risk → dual warmup).

**Estimated Effort**: medium — param regrouping (4 optimizer groups), time-keyed freeze flip, dual-variant warmup, sanity for schedule/freeze/sequence; one gated run (~9 min) + escalation only on a bar-clearing read.

**Risk Assessment**: Branches all terminal: (i) ≥ 96.81 → replicate-pair escalation (mean decides, EXP-052 protocol); (ii) mean-band at family signatures + step ledger showing the expected tail-step surplus → "freeze free but tail steps worthless" — closes the class AND sharpens the tail-conversion law; (iii) low read → layer-1 tail refinement is load-bearing — closes the class with sign; (iv) mid-run recompile signature (one-off ~15–30s stall, ledger collapse) → engineering failure, one fix attempt (warmup cache), else failed; (v) gate/contention → infra relaunch (max 2).

### 2. Per-layer LR structure via norm-adaptive scaling (LARS-style) — documented, not run
**Summary**: Scale each layer's LR by ‖w‖/‖∇w‖ (LARS trust ratio) instead of one global LR.

**Reasoning (and why not the lead)**: The only other never-probed allocation dimension — but the WD-with-BN effective-LR account (used to explain EXP-015) says weight norms already self-equilibrate so that per-layer effective LRs converge to rough uniformity; LARS would mostly re-derive that equilibrium at +1 all-reduce-style norm computation per layer per step (~+0.5–1ms, a priced deferral). LARS's published wins are large-batch (≥4k) regimes. Null-to-negative expected.

**Sources**: goal-learnings EXP-015 entry; model knowledge: LARS (You et al. 2017).

**Estimated Effort**: low-medium.

**Risk Assessment**: σ-coin-flip with a deferral toll; dominated by Idea 1.

### 3. Gradient clipping (global-norm 5.0) — standing documented-weak candidate
**Summary / Reasoning**: Carried from brainstorm-052/053/054 — mechanism vacuous in stable training, heat-reduction at aggressive thresholds (closed axis). Kept on the books; not run.

**Sources**: brainstorm-052 Idea 2.

**Estimated Effort**: trivial.

**Risk Assessment**: Coin-flip on noise; closes nothing.

## Idea Evaluation

- **Evidence strength**: Idea 1 has a CIFAR-validated external anchor (FreezeOut: freeze cost ≈ 0 at fixed epochs) PLUS a project-internal conversion mechanism (fixed-time budget converts the saving to steps) that the anchor could not exploit — the first candidate since EXP-046 with an anchor whose mechanism survives the absorption screen (compute reallocation is not something augmentation can supply). Idea 2's mechanism is pre-refuted by the project's own effective-LR account; Idea 3 is vacuous.
- **Mechanism clarity**: Idea 1's causal chain is explicit and instrumented at every link: freeze → measured dt drop (step ledger will show the tail-step surplus directly) → more tail steps + more plateau evals → higher max. Each link can fail independently and each failure is informative.
- **Expected impact**: honest +0.05–0.15 (optimistic-end bar-clearing); the only candidate on the board whose mechanism could plausibly reach +0.1 without violating a law.
- **Risk profile**: the one real tail risk (charged mid-run recompile) is detectable in the ledger and pre-registered as an engineering branch with one fix attempt. Research branches all fail graceful and terminal.
- **Feasibility**: medium engineering (4 param groups, freeze flip, dual warmup) but every piece uses validated in-project patterns (time-keyed scheduling, warmup extension, composite gates).

Idea 1 dominates — it is the only construction that opens a NEW mechanism class rather than re-measuring a closed one.

## Chosen Idea
**Selected**: Idea 1 — FreezeOut-style tail freezing of stem+stage1 with per-layer heat compensation (FREEZE_FRAC = 0.70)

**Why this idea**:
After 48 nulls closed every catalogued axis, compute ALLOCATION across layers and time is the last untouched dimension — and it carries a CIFAR-validated anchor whose published "no accuracy loss" at fixed epochs becomes "free extra tail steps" under this project's fixed-time budget. No standing law prices the mechanism, and every branch (including the engineering failure) is terminal for the class.

**Hypothesis**:
Freezing stem+layer1 at p = 0.70 (after a heat-preserving compressed anneal for those layers) converts ~22% of late-step compute into ~+1,300 extra tail steps (~+13 epochs) for layers 2–3 plus ~+13 extra plateau evals; if FreezeOut's freeze-cost ≈ 0 transfers, best_test_acc reads above the recipe mean and ≥ 96.81 if the tail-conversion is worth ≥ +0.3 (one-draw detectable). Pre-registered branches: (i) read ≥ 96.81 → replicate-pair escalation (improvement iff MEAN of two ≥ 96.81); (ii) mean-band [96.41, 96.73] at family-shaped signatures WITH the expected step surplus visible in the ledger → freeze free but tail steps sub-σ; class closed, tail-conversion law sharpened; (iii) read < 96.41 → layer-1 tail refinement is load-bearing (parameter-side tail-pressure law established alongside the data-side one); class closed with sign; (iv) mid-run recompile signature (one-off ≥ 15s stall in watchdog windows / step ledger far below family with clean dt) → engineering failure, ONE fix attempt (dual-warmup cache), else Outcome failed; (v) GATE_KILL/contention → infra relaunch (max 2). Expected signatures: D0 22.3–22.8ms pre-freeze, windowed dt dropping to ~17.5–18.5ms post-freeze (the direct mechanism check), epochs ≥ 145, params 4,286,026.
