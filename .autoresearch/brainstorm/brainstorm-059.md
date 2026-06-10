# Brainstorm EXP-059
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- No new external sources. This loop operates entirely within the heavily-mapped project space; the binding evidence is the experiment history, not new literature. (AugMix ICLR 2020 and the GPU grid_sample primitives from EXP-056/057 are the relevant prior art, already in-context.)

## Experimental History Review
- **Current best 96.45** (EXP-054): `RandomApply([AugMix() w3,d-1], p=0.5)` + GPU Cutout(16), k=4 ResNet-20. Augmentation chain-COUNT diversity is the ONLY lever that has ever lifted top-1 (96.00→96.45 over EXP-012/052/054).
- **Augmentation axis FULLY EXHAUSTED**: magnitude (EXP-053), width>3 (EXP-055), coverage<50% (EXP-055), coverage=100% GPU faithful (EXP-057), naive-harsh-GPU (EXP-056), policy-swap (EXP-014: TA≈RA≈AugMix), mixing (EXP-011/018), cooldown (EXP-033/034/035), border-quality (EXP-037), occlusion strength/pattern (EXP-013/021/048). The winner already uses chain_depth=-1. ~50% coverage is a TRUE interior optimum (EXP-055 below / EXP-057 above).
- **Capacity axis CLOSED from ALL FOUR directions** (this loop's update): uniform widen (EXP-004/009), FLOP-neutral realloc (EXP-038), deeper-narrower (EXP-044), shallower-wider (EXP-058). The wide-conv memory-bandwidth wall is set by channel WIDTH not block-count; k=4 ResNet-20 is definitively the compute-optimal frontier.
- **Also closed**: optimizer family+grad+objective (EXP-030/031/036/041/043), weight-averaging EMA/SWA (EXP-006/019/020, count≥3 High), LR-schedule peak+floor+shape (EXP-016/017/019/020/029), normalization (EXP-047), activation (EXP-010/028), classifier-head (EXP-032/039), residual-scaling (EXP-026/051), batch-size (EXP-025/050), dropout (EXP-022), throughput→epochs (EXP-007/045/046, epoch-saturated at ~91), cheap-throughput flags (EXP-040), downsampling (EXP-024/027), near-miss combination (EXP-049).
- **Key structural fact (re-derived)**: CPU augmentation is FREE w.r.t. the Σdt/epoch budget (parallel dataloader workers) — it only costs WALL (600s limit). The GPU step dt (8ms) caps epochs at ~91 independent of CPU aug. EXP-054's 50% coverage was a WALL limit, not an epoch limit. The GPU-aug path (EXP-056/057) trades epoch-cost for unlimited coverage — and full coverage underfit.
- **Genuinely-remaining untested**: (a) GPU faithful AugMix at the PROVEN 50% coverage (only EXP-056/057 100%-coverage GPU runs exist; 50% is untried and is where the proven optimum lives); (b) WARMUP_FRAC (the one LR-schedule sub-knob never isolated); (c) scale/zoom jitter (the one common geometric primitive ABSENT from AugMix's op pool).

## Candidate Ideas

### 1. GPU faithful AugMix at the proven p=0.5 coverage (W=3) — the one remaining same-family augmentation variant
**Summary**: Reuse the EXP-057 `gpu_augmix` primitives (W=3 Dirichlet multi-chain affine+photometric + Beta clean-mix, GPU-side) but apply to a random ~50% of each batch (the proven-optimal coverage), the other ~50% crop+flip+Cutout only. At 50% coverage the grid_sample cost roughly halves vs EXP-057's 100% → expect dt ~9ms → ~84 ep (within budget). Removes CPU AugMix; CPU → crop+flip+ToTensor+Normalize.
**Reasoning**: Builds on the ONLY proven lever (augmentation diversity) at its PROVEN operating point (50% coverage = EXP-054 winner), using the validated-cheap GPU infra (EXP-056). The GPU path delivers CONTINUOUS-magnitude affine chains (rotation/shear/scale) vs torchvision AugMix's discrete PIL ops — potentially richer per-sample diversity at the same coverage. Resolves the open question (does the GPU path match/beat CPU AugMix at the proven coverage?).
**Sources**: EXP-054 (w3/50%=96.45), EXP-056 (GPU infra validated cheap), EXP-057 (gpu_augmix code + clean-mix), goal-learnings § augmentation.
**Estimated Effort**: low-medium (port gpu_augmix from EXP-057 git history + add a per-batch 50% mask; dt/epoch gate; idle-GPU launch).
**Risk Assessment**: LOW risk, LOW-MODERATE ceiling. By construction ≈ the EXP-054 winner (96.45); the continuous-affine "richer chains" claim is the only path above the bar and EXP-014 warns policy/op-set rarely matters once strong diverse aug is present. Likely lands within the ±0.25pp jitter of 96.45. Graceful failure (no-improvement near baseline). Diagnostic value (GPU-vs-CPU parity) is real regardless.

### 2. WARMUP_FRAC sweep (0.05 → 0.10) — the one untested LR-schedule sub-knob
**Summary**: Raise `WARMUP_FRAC` 0.05→0.10 (linear warmup over 10% of the time budget instead of 5%). Single-constant, compute-neutral.
**Reasoning**: EXP-016/017 swept PEAK_LR, EXP-019/020 the floor, EXP-029 the shape — but warmup LENGTH was never isolated. At a short ~91-ep budget, warmup length trades early-stability against anneal time. The one LR-schedule cell never probed.
**Sources**: EXP-000 (set 0.05 heuristically), EXP-016/017/019/020/029 (rest of schedule axis).
**Estimated Effort**: trivial (one constant).
**Risk Assessment**: LOW ceiling — the LR-schedule axis is declared closed; warmup-frac is a minor knob and BN+LS already stabilize early steps, so 0.05 is likely near-optimal. Likely within-noise null. Cheap, clean, can't epoch-wall.

### 3. Scale/zoom jitter — the one geometric primitive absent from AugMix's op pool
**Summary**: Replace `RandomCrop(32, padding=4)` with `RandomResizedCrop(32, scale=(0.7,1.0), ratio=(0.9,1.1))` (CPU, free w.r.t. epochs), adding object-scale diversity that AugMix's pool (shear/translate/rotate/color) lacks. Keep AugMix-50% + Cutout.
**Reasoning**: Scale/zoom is a genuinely-new input-diversity axis not covered by any prior aug experiment; the augmentation lever is the only proven plateau-breaker, so a genuinely-new primitive is the most on-mechanism long-shot.
**Sources**: EXP-012/052/054 (aug is the lever), EXP-014 (policy saturation caveat).
**Estimated Effort**: low (one transform swap).
**Risk Assessment**: MODERATE regression risk — stacking scale jitter on the already-saturated AugMix+Cutout recipe may over-regularize → underfit at 91 ep (the EXP-011/018/022 failure mode). EXP-014 also warns added ops rarely help once diverse aug is present. Could regress rather than gently null.

## Idea Evaluation
The plateau is exhaustively mapped; every remaining option is a low-ceiling long-shot, so the choice is governed by evidence strength (the skill's primary criterion) and graceful-failure profile. Candidate 3 (scale jitter) has the most plausible "new diversity" mechanism but the worst risk profile (over-regularization regression on the saturated recipe) and EXP-014 argues against it. Candidate 2 (warmup) is the cleanest/cheapest but sits on a declared-closed axis with a weak mechanism. Candidate 1 (GPU AugMix at 50%) has by far the strongest evidence base — it operates the ONLY proven lever at its PROVEN-optimal coverage using validated infra, has the safest failure mode (≈baseline, no over-regularization), and resolves a genuinely-open question (GPU continuous-affine vs CPU discrete AugMix at the proven operating point), with a small real chance the richer continuous chains edge above the bar.

## Chosen Idea
**Selected**: Candidate 1 — GPU faithful AugMix at the proven p=0.5 coverage (W=3).

**Why this idea**: With augmentation sub-axes and the entire capacity/optimizer/schedule/normalization map closed, the highest-evidence remaining move is to operate the one proven lever (augmentation diversity) at its proven-optimal 50% coverage via the validated GPU path — the single same-family variant never run. It has the safest failure mode and resolves the open GPU-vs-CPU parity question, with a modest real chance the continuous-affine chains beat torchvision's discrete ops.

**Hypothesis**: GPU faithful AugMix (W=3 Dirichlet + Beta clean-mix) on a random ~50% of each batch will hold dt ≈ 9ms (epochs ≥ ~82) and wall < 600s, and — because it reproduces the EXP-054 winner's proven coverage while delivering continuous-magnitude affine chains potentially richer than torchvision's discrete ops — best_test_acc ≥ 96.55 (bar). A within-noise result near 96.45 would confirm the GPU path merely matches CPU AugMix (parity established, no headroom from coverage/representation); a regression would indicate the continuous-affine chains are harsher than torchvision's discrete pool at matched coverage.
