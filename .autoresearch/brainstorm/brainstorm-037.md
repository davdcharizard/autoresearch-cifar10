# Brainstorm EXP-037
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Squeeze-and-Excitation Networks (Hu, Shen, Sun, CVPR 2018, arXiv 1709.01507)**: channel attention via global-average-pool → bottleneck MLP (reduction r=16) → sigmoid → channel-wise rescale, inserted after the last BN of each residual block before the add. **In-domain evidence**: the paper's CIFAR-10 experiments report ResNet-110 error 6.37 → 5.21 (≈ +1.16 accuracy) at fixed epochs; SE consistently adds +0.5–1.2 on CIFAR ResNets across depths at <1% params/FLOPs. This is the largest published in-domain effect size of any mechanism class not yet tried in this loop, and it is a converged-LEVEL mechanism (a new functional capability — input-dependent channel reweighting — not a transit accelerant).
- **Identity Mappings in Deep Residual Networks (He et al., ECCV 2016, arXiv 1603.05027)**: pre-activation block ordering (BN-ReLU-conv) improves optimization via clean identity paths; CIFAR gains concentrate at extreme depth (110–1001 layers), near-zero or negative at depth ~20.
- **ECA-Net (Wang et al., CVPR 2020)**: SE variant replacing the MLP with a 1D conv; comparable gains, fewer params — fallback engineering option if SE's small matmuls price badly on H20.
- Knowledge base (knowledge/README.md): no existing attention entry; cifar10-fast/airbench anchor rows note neither uses attention (their budgets are ~10 epochs where SE's extra launches do not amortize). A distilled SE note will be added at plan time.

## Experimental History Review

- 38 experiments, 6 improvements (last EXP-006); baseline 96.71 @ 1990397 (recipe mean ≈96.57, σ ≈0.16); bar 96.81; **31 consecutive misses**.
- **Recipe space audit-complete as of EXP-036**: every constant in train.py individually dosed; incumbent won or tied every time (LS read exactly flat at the anchor dose). Axes measured-closed: recipe constants, schedule (family/heat/warmup), optimizer (internal + geometry), gradient noise (bracketed both directions, implicit AND explicit — EXP-035 SAM), data pressure (dose-response peaked, time-varying doses lose both directions), eval-side BN manipulation (EXP-029 −10.9), capacity (width magnitude/allocation/depth both ways/head), init (both directions), throughput (numerics-equivalence law), loss-target (EXP-036).
- **What remains by elimination**: architecture changes adding NEW FUNCTIONAL CAPACITY while free in the four currencies (early heat, epochs/dt, numerics, noise). The structural-law screens an SE candidate must pass: (a) DEFERRAL — no component may need learning during peak heat from a cold init (EXP-018 zero-γ −0.99; EXP-020 projections −0.13) → SE must start near-identity; (b) dt — H20 is launch-bound (EXP-034: +2.5ms/block width-independent; EXP-026: pointwise special functions 1–4.5ms) → SE's ~6 small ops/block must be priced by an early-dt gate, with kill threshold; (c) max-statistic — only converged plateau LEVEL pays; SE's published gain is a level claim; (d) noise — SE leaves batch/momentum/aug untouched.
- σ discipline: candidate TRUE effect must be ≥ +0.3. SE's in-domain published effect (+0.5–1.2) is the first untried mechanism that clears the screen even after paying a 1–2ms dt deficit (−0.1…−0.2 by the linear law). External transfer is 0-for-13 — but every prior transfer failed via a now-named law (deferral, max-statistic, dt, noise, augmentation-regime mismatch), and this candidate is explicitly engineered against each.

## Candidate Ideas

### 1. SE channel attention with near-identity init (r=16, all 9 blocks)
**Summary**: Insert the standard SE module (adaptive_avg_pool → Linear(C, C/16) → ReLU → Linear(C/16, C) → sigmoid → channel-wise multiply) after bn2, before the residual add, in all 9 BasicBlocks. **Near-identity init**: fc2 bias init to +2.0 so sigmoid ≈ 0.88 at step 0 — the block starts ≈ baseline-scaled (BN downstream of the multiply does not exist, but the residual add + final ReLU see a 0.88-scaled branch, a mild perturbation vs sigmoid(0)=0.5 halving that would replay EXP-018's deferral). Params +~33k (+0.8%); same block count, widths, batch, schedule, optimizer.

**Reasoning**: The only mechanism class untried in 37 experiments whose published in-domain effect (+0.5–1.2 on CIFAR ResNets at fixed epochs, Hu et al. Table on CIFAR-10) survives the σ screen after deficit arithmetic. Mechanism is a LEVEL mechanism: input-conditioned channel gating is a capability the static recipe cannot emulate — it changes the converged function class, not the trajectory speed. Engineered against every named law: near-identity init (deferral), early-dt gate with pre-registered kill (launch-bound pricing: estimate +1–3ms from ~6 extra kernels/block under default compile fusion; kill ≥27ms), numerics unchanged (default compile, bf16), noise untouched. Net arithmetic at +2ms: −12 epochs ≈ −0.17 deficit; published gain ≥ +0.5 → net ≥ +0.3 ≈ bar.

**Sources**: arXiv 1709.01507 (CIFAR table); goal-learnings Patterns (deferral, dt, max-statistic, noise laws); project-insights (launch-bound hardware law EXP-005+034; pointwise pricing EXP-026).

**Estimated Effort**: medium — ~25 lines in BasicBlock + init; standard composite run with dt gate.

**Risk Assessment**: (a) dt overrun is the dominant risk — if SE prices ≥ +4ms the deficit eats the published gain; mitigated by gate-kill within 2 minutes (cheap information: prices channel attention on H20 permanently, and ECA remains as a cheaper re-try); (b) deferral residue even at sigmoid≈0.88 — visible in ep1–5 evals vs family (38±1), diagnosable; (c) transfer failure (SE gain could be calibrated to longer/era-2018 recipes without TA/RE — the EXP-035/036 absorption law may extend to attention); that outcome closes the attention axis with a measured datum.

### 2. Pre-activation block reordering (He et al. 2016)
**Summary**: Reorder BasicBlock to BN→ReLU→conv→BN→ReLU→conv with clean identity add; final BN+ReLU before pool. Same op count → dt ≈ baseline (±0.5ms from the one extra stem/final BN).

**Reasoning**: Free in dt/params and famously robust — but the published CIFAR gains concentrate at depth 110–1001 where identity-path degradation actually binds; at depth 20 the original paper shows pre-act ≈ post-act (sometimes worse). Expected effect at our depth ~0±0.2 — fails the σ screen. Also perturbs the certified BN/heat calibration for near-zero expected payoff.

**Sources**: arXiv 1603.05027 §4.2 (CIFAR depth ablations).

**Estimated Effort**: low-medium.

**Risk Assessment**: Sub-σ expected effect; the certified-recipe perturbation risk (EXP-029-class BN coupling) is asymmetric against it.

### 3. Classifier-head damping (fc zero-init or cifar10-fast logit ×0.125 temperature)
**Summary**: Zero-init the final Linear (uniform logits at step 0) or scale logits by 0.125 in forward (CE temperature 8, anchor: cifar10-fast).

**Reasoning**: Zero-dt and one-line, flagged in exp-report-035/036 Next Steps — but it now stands against TWO measured laws: the init law ("at ~139 epochs nothing init-time moves the converged plateau", EXP-018/019 both directions) kills the init variant's mechanism, and the EXP-036 LS null plus the anchor-transfer caveat (augmentation-regime mismatch; cifar10-fast pairs the temperature WITH LS 0.2 at ~10 epochs) kill the temperature variant's. Expected effect ≈ 0.

**Sources**: cifar10-fast (knowledge README); exp-report-036.md; goal-learnings init entries.

**Estimated Effort**: low.

**Risk Assessment**: Safest failure mode but near-zero expected effect — a slot-filler, not a candidate to beat a +0.3 screen.

## Idea Evaluation

Evidence strength: Candidate 1 dominates — the SENet paper's CIFAR-10 ResNet numbers are the largest in-domain published effect of any untried mechanism (+1.16 at depth 110; conservatively +0.3–0.5 at depth 20/4× wide), where Candidates 2 and 3 are both expected-zero BY THE PROJECT'S OWN MEASURED LAWS (depth-20 pre-act ablations; init law + LS null + anchor-transfer caveat). Mechanism clarity: SE adds input-conditioned channel gating — a strictly larger function class converging to a different (per literature, better) plateau; clear LEVEL mechanism aligned with the max-statistic law. Expected impact: only Candidate 1 clears the σ screen after deficit arithmetic. Risk: Candidate 1's dominant failure mode (dt overrun) is gate-killable within 2 minutes at ~zero cost, and its full-run failure mode is a clean no-improvement that closes the attention axis; nothing destabilizing. Feasibility: ~25 lines, all in train.py. Candidate 1 wins on every criterion that matters; 2 and 3 are recorded as screened-out so future loops do not re-derive them.

## Chosen Idea
**Selected**: SE channel attention with near-identity init (Candidate 1)

**Why this idea**:
After EXP-036 completed the recipe-constant audit, the only open territory the laws permit is architecture that adds new functional capacity while staying free in early heat, dt, numerics, and noise. SE is the best-evidenced such mechanism in the literature for exactly this dataset and architecture family, its known costs map one-to-one onto laws this project has already learned to engineer around (near-identity init for deferral, early-dt gate for launch-bound pricing), and its published effect size is the first in 13 external-transfer attempts large enough to survive both the σ screen and the deficit arithmetic.

**Hypothesis**:
SE modules (r=16, fc2 bias +2 near-identity init) in all 9 blocks raise the converged plateau LEVEL via input-conditioned channel reweighting, at a measured dt cost ≤ ~2.5ms (≤ −0.2 deficit), predicting best_test_acc ≥ 96.81. Falsified by (a) GATE_KILL: dt ≥ 27ms within the first 2 minutes — channel attention is unaffordable on launch-bound H20 (ECA becomes the only retry); or (b) a clean converged plateau ≤ baseline band — the heavy-augmentation absorption law (EXP-035/036) extends to attention, closing the axis.
