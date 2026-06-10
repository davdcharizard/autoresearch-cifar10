# Report EXP-044: Depth↔width reallocation — deeper-narrower iso-param ResNet (ResNet-32, k=3)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-044.md
- **Plan**: plans/plan-044.md
- **Log**: logs/exp-log-044.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within a fixed 300s single-GPU training budget by editing only `train.py`. Baseline: **96.22%** (EXP-012, commit 6c417a4). Success bar: **96.32%** (baseline + 0.1pp).

## Idea & Hypothesis
**Chosen idea**: Reallocate the ~4.3M-param budget from WIDTH into DEPTH — a deeper, proportionally-narrower iso-param ResNet. Across all 45 prior experiments, every capacity change manipulated *width* (EXP-001 k=4, EXP-004 k=6, EXP-009 k=5, EXP-038 stage realloc); the number of blocks per stage stayed fixed at 3 (ResNet-20 depth). The "capacity closed (width/depth)" claim (EXP-038) asserted depth by analogy to width but had no depth datapoint. Depth is mechanistically distinct and is the founding ResNet paper's own CIFAR scaling lever (He 2016: ResNet-20→110 improved monotonically at fixed narrow width) — a genuine generalization/inductive-bias lever, the class the polish-wall says is required for top-1 gains.

**Hypothesis**: A deeper-narrower net at iso-param (5 blocks/stage, k=3, {48,96,192}, ResNet-32) sized for ≈iso-FLOP would, if dt stayed ≈8ms and epochs ≥77 (EXP-007 saturation), generalize better per parameter than the wide-shallow k=4 baseline and clear 96.32. The null/regression case: added sequential layers raise dt enough to drop epochs below saturation (underfit), and/or depth's per-param advantage does not materialize on this already-well-trained 32×32 net.

## Approach
Single code change in `train.py` (L19-20): `NUM_BLOCKS` 3→5 and `WIDTH_MULT` 4→3. This converts the baseline ResNet-20 (3 blocks/stage @ {64,128,256}, 4.30M params, 9 blocks) into a ResNet-32 (5 blocks/stage @ {48,96,192}, 4.17M params, 15 blocks). Pre-launch param/FLOP math (validated by reproducing the exact baseline count 4,299,866) predicted 4,166,970 params (96.9%) and 605M FLOPs (97.8%) — both slightly *below* baseline, all widths multiples of 16 (tensor-core safe). The entire recipe (SGD+Nesterov, time-fraction cosine LR peak 0.2, TA+Cutout(16), label smoothing 0.1, bf16, channels_last, torch.compile reduce-overhead, seed 42) was byte-identical → clean single-variable depth-vs-width test. Ran on idle GPU 1 (GPU 0 was busy with another user's job; GPU 1 verified idle before launch and after run).

## Execution
One clean run, no retries, no errors/NaN/OOM. Banner confirmed `ResNet-32 | params: 4,166,970` exactly as predicted. Completed in 380.4s total wall (300.0s timed training + 2.1s startup). dt was **rock-steady at 12ms** from step 50 through step 23200 (460/464 sampled steps = 12ms) — a flat architectural floor, definitively NOT contention (which spikes; GPU 1 was idle throughout, both GPUs 0% after). Reached only **60 epochs / 23,243 steps** (vs baseline ~91 / ~35k). Per-epoch convergence was also visibly slower: ep1 20.7%, ep2 30.2%, ep3 32.4% (vs baseline ~55% by ep1).

## Results
- **Primary metric**: best_test_acc **92.58%** (baseline 96.22, delta **−3.64pp**, −3.78%). final_test_acc 92.55%, final_test_loss **0.2905** (≫ baseline 0.195). peak_vram 569MB.
- **Observations**: The decisive finding is throughput, not generalization: despite ≈iso-FLOP (97.8%), the deeper-narrower net cost **dt 8→12ms (+50%)** — *worse* than EXP-038's fat-head width realloc (+31%). Root cause: 15 sequential conv+BN blocks (vs 9) at lower per-layer arithmetic intensity → memory-bandwidth/launch-bound (more kernels, each doing less useful work; CUDA-graph replay amortizes launch overhead but not the per-kernel execution + memory traffic). This dropped the run to 60 epochs, far below the ~77-epoch saturation point → severe underfit (loss 0.29, still falling-ish at ep60), compounded by the deeper net's slower early optimization.
- **Analysis**: The hypothesis is *not cleanly testable at this budget* — the fairness gate failed (dt 12ms, 60ep). We cannot separate "depth doesn't help generalization here" from "the net never trained enough." But the experiment decisively answers the *practical* question: deeper-narrower at iso-param/iso-FLOP is NOT dt-neutral on this hardware — it blows the dt budget harder than any prior capacity move, so depth cannot be added without triggering the epoch wall. This empirically confirms (rather than merely asserts) EXP-038's "capacity axis closed for depth too" — the depth direction hits the *same* monotone epoch wall as width (project-insights High: any compute-/sequential-layer-adding change underfits at 300s), and in fact a steeper one. Fits the trajectory: 34 consecutive no-improvements since EXP-012; the 96.22 k=4/300s plateau is the robust ceiling across width AND now depth.
- **Key Learning**: Deeper-narrower at iso-FLOP is *not* dt-neutral — 15 sequential blocks raised dt 8→12ms (+50%, worse than width realloc) → 60ep≪77 → severe underfit (−3.64pp); the depth axis hits the same (steeper) epoch wall as width.

## Verification
- **Conditions**: condition 3 (primary metric ≥ 96.32) FAILED (92.58). Conditions 1 (clean completion within budget) and 4 (no scope/constraint violations — only train.py changed) PASSED. Condition 2 (fairness gate) FAILED (dt 12ms, 60ep) → result is dt-confounded.
- **Review Notes**: Results trustworthy. dt = 12ms verified architectural (flat across all 23k steps, GPU idle before/after, no contention spikes), not an infra artifact. The −3.64pp is a genuine underfit (loss 0.29 ≫ 0.195, slow early convergence), consistent with 60 epochs. No false pass/fail concerns; scope clean (`git diff --name-only` = train.py only).
- **Verdict**: no-improvement
- **Verdict Basis**: primary necessary condition failed (92.58 < 96.32); run was valid, on-scope, and completed cleanly (not invalid/crash). dt-confounded underfit.

## Unexplored Avenues
- **Milder depth increase (ResNet-26, 4 blocks/stage, k≈3.5 {56,112,224})**: would cost less dt than +50%, but the trend (each block ≈ +0.5-0.7ms; EXP-038 fat-head already hit +31%/73ep at iso-FLOP) predicts ~10-10.5ms → ~73ep → still underfit like EXP-038. Low chance of a *fair* test; the dt-vs-depth curve is unfavorable.
- **Depth added ONLY in the cheapest-to-execute stage**: per-block FLOPs are ≈equal across stages, but per-block *wall-clock* differs by memory pattern; no stage is cheap enough at this hardware floor to add blocks dt-neutrally (EXP-038 showed even stage3 realloc is memory-bound). Unlikely to escape the wall.
- **Depth WITHOUT narrowing (pure ResNet-32 at k=4, ~7M params)**: would add both FLOPs and layers → far worse epoch wall (cf. k=5/k=6 EXP-004/009). Not viable.
- The honest read: depth, like width, is gated by the 300s epoch wall — the whole CAPACITY axis (width + depth + realloc) is now empirically closed.

## Next Steps
- **Document the 96.22 plateau as the robust k=4/300s ceiling (confidence: high)**: capacity is now closed from all three directions (width EXP-004/009, FLOP-neutral width realloc EXP-038, depth EXP-044), joining the already-closed augmentation, schedule, regularizer, optimizer-family, objective, weight-averaging, classifier-head, and intermediate-feature-routing axes — 34 straight no-improvements.
- **The ONLY remaining lever is a genuine dt REDUCER that buys net-new epochs at k=4 (confidence: low-medium)**: every capacity/generalization lever is gated by the epoch wall, so the unblock must come from throughput. EXP-040 closed cheap flags (cudnn.benchmark no-op; conv dt floor reached under reduce-overhead). Untried: `torch.compile(mode="max-autotune")` (aggressive kernel autotune — risk: compile tax inside the timed budget) or moving augmentation fully off the timed GPU path. If a real dt reduction materialized (e.g. 8→6ms → ~120 epochs), it could re-open whether the net is epoch-saturated or convergence-bound at a higher epoch count — the one genuinely-open question (project-insights EXP-040 corollary).
- **Radical non-capacity architecture (confidence: low)**: ResNeXt grouped convs remain formally untried but carry the highest dt-confound risk (grouped convs are memory-bound on H20 — worse than this depth result); deprioritized.

## Exit Action Results
- None defined for this goal — skipped.
