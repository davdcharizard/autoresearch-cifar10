# Brainstorm EXP-044
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Deep Residual Learning for Image Recognition** (He et al., CVPR 2016; the in-code ResNet citation, train.py L61): the original CIFAR experiments scaled ResNet by DEPTH (20→32→44→56→110 layers, 3/5/7/9/18 blocks per stage) at the SAME narrow width {16,32,64}, and accuracy improved monotonically with depth (ResNet-56 ≈ 93.0%, ResNet-110 ≈ 93.6% vs ResNet-20 ≈ 91.3%). Depth, not width, was the original CIFAR scaling lever; deeper-narrower nets are more parameter-efficient per generalization than shallow-wide ones at equal params.
- **Wide Residual Networks** (Zagoruyko & Komodakis, BMVC 2016; knowledge/papers/wrn-dropout.md): WRN showed width is a *faster-to-train* substitute for depth, but the depth/width trade-off is a frontier — at equal parameter budget, the optimal point depends on the compute/epoch budget. Our net sits at the extreme wide-shallow corner (k=4, only 3 blocks/stage = ResNet-20 depth). The depth direction of this frontier is unmapped here.
- **project-insights.md** (local, High Importance): the epoch-wall entry (EXP-004/009/015/024/036/038) — ANY compute-adding change underfits at 300s; iso-FLOP changes (EXP-015 pre-act, EXP-038 realloc) STILL lost epochs because the compiled graph/memory pattern shifted. EXP-007: the net is epoch-SATURATED past ~77 epochs (we currently get ~91), so there is ~14 epochs of slack to absorb a modest dt increase before underfitting begins.

## Experimental History Review

- **Current best / baseline**: 96.22% (EXP-012, commit 6c417a4), k=4 WideResNet ({64,128,256}, 4.30M params), 3 blocks/stage (= ResNet-20 depth), ~91 epochs @ 8ms/step. Plateau confirmed robust: 33 consecutive no-improvements since EXP-012.
- **What worked (6 improvements)**: modern recipe (EXP-000), WRN width k=4 (EXP-001, the dominant lever, +2.84pp), Cutout (EXP-002), GPU-vectorized Cutout (EXP-003), TrivialAugment (EXP-012). All capacity gains came from WIDTH.
- **Closed axes** (all no-improvement): augmentation (strength/policy/mixing/cooldown/border — EXP-011/013/014/018/021/033-035/037), LR schedule (peak+floor+shape — EXP-016/017/019/020/029), regularizer-adds (WD/dropout — EXP-005/022), classifier head (EXP-032/039), intermediate-feature-routing (EXP-032/042), activations (EXP-010/028), weight-averaging (EXP-006/019/020), optimizer family + grad/objective mods (EXP-030/031/036/041/043), bag-of-tricks (EXP-026), large-batch (EXP-025), cheap-throughput (EXP-040).
- **CAPACITY — what was actually tested**: k=6 (EXP-004), k=5 (EXP-009) — both wider, compute-bound, underfit. Fat-head stage realloc {48,128,304} (EXP-038) — FLOP-neutral but memory-bound, dt 8→10.5ms, underfit. **All three changed WIDTH. The number of blocks per stage (DEPTH) was NEVER changed — it has stayed at 3 (ResNet-20) for all 45 experiments.** The "capacity closed from both directions (width/depth)" claim (EXP-038, the k=4/k=6 entry) asserts depth by analogy to width; it has no depth datapoint.
- **Untried gap**: the depth↔width trade-off at iso-param. This is the single structural lever with both (a) a strong literature prior (depth > width per-param on CIFAR, He 2016) and (b) zero prior coverage here.

## Candidate Ideas

### 1. Depth↔width reallocation — deeper, proportionally-narrower iso-param ResNet
**Summary**: Hold the ~4.3M parameter budget and the 300s budget fixed, but move capacity from width into DEPTH: increase blocks-per-stage from 3 (ResNet-20) to 5 (ResNet-32) and reduce the width multiplier to keep params ≈ 4.3M (e.g. NUM_BLOCKS=5 with k≈3 → {48,96,192}, or an asymmetric variant that adds the extra blocks preferentially in stage2/stage3 where each block costs the same FLOPs as a stage1 block — per-block FLOPs ≈ C²·area is ≈ constant across stages since C doubles as area quarters). Tune the exact (NUM_BLOCKS, per-stage widths) so that (i) all channel counts stay multiples of 16 for the tensor-core path (EXP-038 lesson), (ii) total params land within ±2% of 4.30M, and (iii) measured dt stays close to 8ms so realized epochs stay ≥77 (the EXP-007 saturation point) — leaving the depth-generalization benefit unconfounded by the epoch wall.

**Reasoning**: Depth is the one capacity dimension never tested, and it is the dimension the founding ResNet paper used to scale CIFAR accuracy (ResNet-20→110 monotonically improved at fixed narrow width). The polish-wall (project-insights Medium) is explicit that "top-1 gains require capacity or fundamentally different generalization, not optimization/objective polish" — depth changes the inductive bias / effective receptive field / feature-hierarchy granularity, a genuine generalization lever, not polish. Two project-specific factors make a fair test plausible: (1) EXP-007 shows ~14 epochs of slack above the ~77-epoch saturation point, so a modest dt rise need not underfit; (2) under torch.compile(reduce-overhead) the whole forward is one replayed CUDA graph, so added layers add kernel *execution* time but not per-launch overhead — and a deeper-NARROWER net at iso-FLOPs may be closer to dt-neutral than the wider nets (EXP-004/009) that strictly added FLOPs.

**Sources**: He et al. 2016 (train.py L61 citation); knowledge/papers/wrn-dropout.md (WRN depth/width frontier); goal-learnings "Widening past k=4" (EXP-004/009) and "fat-head realloc" (EXP-038) — both width-only; project-insights epoch-wall (High) + EXP-007 saturation + EXP-038 tensor-core-alignment note.

**Estimated Effort**: low — only NUM_BLOCKS (L19) and WIDTH_MULT (L20) / the _make_layer width arithmetic change; no new ops, no new deps, recipe untouched.

**Risk Assessment**: The dominant risk is the epoch wall: more sequential conv+BN layers raise dt (EXP-015 lost 91→78 epochs from mere block REORDERING; a genuinely deeper net adds real layers) → fewer epochs → underfit → dt-confounded regression rather than a clean depth test. Narrowing also risks dropping arithmetic intensity / tensor-core efficiency (EXP-038: non-multiple-of-8 widths ran ~5× slower). Mitigations: size for ≈iso-FLOP and verify dt≈8ms and epochs≥77 before trusting any delta; keep all widths multiples of 16. Worst case: a clean no-improvement that finally CLOSES the depth axis empirically (currently only asserted) — still informative.

### 2. Progressive resizing — cheap low-resolution early epochs, full-resolution finish
**Summary**: Train the first ~60–70% of the budget on 16×16 down-sampled inputs (F.interpolate of the 32×32 augmented batch on GPU), then switch to full 32×32 for the remaining tail and the final convergence; eval is always at 32×32 (frozen). Lower-resolution convs are ~4× cheaper in FLOPs, which — if any of the 8ms/step is compute-bound — buys substantially more early epochs/updates for fast feature learning, with the full-res tail aligning the model to the eval distribution.

**Reasoning**: The recurring failure mode is "generalization-bound at fixed capacity but cannot add compute"; progressive resizing is the one lever that could *add effective epochs without adding net compute*. fast.ai popularized it for ImageNet. It attacks the budget constraint directly rather than the capacity constraint.

**Sources**: fast.ai progressive resizing; project-insights cheap-throughput entry (EXP-040) — the conv dt floor question.

**Estimated Effort**: medium — resolution schedule + a GPU resize in the train loop; CUDA-graph shape changes force a recompile at the switch point (manageable but adds compile tax inside the timed region).

**Risk Assessment**: HIGH uncertainty on the core premise — EXP-040 found the net is at the "conv dt floor" (largely launch/overhead-bound under reduce-overhead), so halving resolution may NOT cut dt much, killing the whole rationale. CUDA-graph recompile at the resolution switch adds dt inside the budget. 16×16 CIFAR is very coarse (loses fine texture the eval needs). Several ways to fail; lower confidence than #1.

### 3. ResNeXt-style grouped (cardinality) convolutions at iso-param
**Summary**: Replace the 3×3 convs in BasicBlock with grouped convs (cardinality C, e.g. groups=8 or 16) plus widened bottleneck channels to keep params ≈ 4.3M — the ResNeXt "aggregated transformations" architecture, which improves ImageNet accuracy at equal params/FLOPs vs plain ResNet.

**Reasoning**: A genuinely different architecture FAMILY (grouped/cardinality) rather than a scalar knob; ResNeXt's split-transform-merge gives better accuracy-per-FLOP on ImageNet.

**Sources**: ResNeXt (Xie et al. 2017); flagged as the "radical iso-dt gamble" in the EXP-043 report Next Steps.

**Estimated Effort**: medium — restructure BasicBlock to grouped convs + channel bookkeeping.

**Risk Assessment**: HIGHEST dt-confound risk. Grouped convs are notoriously memory-bandwidth-bound and poorly tensor-core-optimized on GPUs — EXP-038 already showed memory-bound restructures blow up dt (8→10.5ms) despite ≈iso-FLOPs. Very likely to lose epochs and regress for dt reasons, masking any architectural merit (same trap as #1 but with worse expected dt behavior and weaker per-param generalization prior on small images than depth). Deprioritized relative to #1.

## Idea Evaluation

All three are "radical architecture" gambles, appropriate now that every incremental axis is closed (33 straight no-improvements) and the directive is to try more radical changes. They differ sharply on evidence strength, mechanism clarity, and dt risk.

- **Evidence strength**: #1 (depth) has the strongest and most directly-comparable prior — the *founding ResNet paper's own CIFAR experiments* scaled by depth and improved monotonically at fixed width; depth > width per-param is a textbook CIFAR result. #3 (ResNeXt) has strong ImageNet evidence but weak small-image/shallow-net evidence (the EXP-027/028 "ImageNet tricks don't transfer to 32×32 shallow nets" pattern applies). #2 (progressive resizing) has the weakest evidence for *this* setup — its premise (resolution↓ ⇒ dt↓) is directly challenged by EXP-040's conv-dt-floor finding.
- **Mechanism clarity**: #1 is clear — depth adds effective receptive field / hierarchy granularity / generalization at iso-param. #2's mechanism (more effective epochs) is clear in principle but its enabling assumption is shaky here. #3's mechanism (cardinality) is clear but its benefit on shallow 32×32 nets is doubtful.
- **Expected impact / risk profile**: all three share the epoch-wall failure mode, but #1 has the best chance of a *fair* test because (a) it can be sized for ≈iso-FLOP and dt-neutrality, (b) it exploits the ~14-epoch slack above saturation, and (c) deeper-narrower may even be dt-favorable vs the width adds that failed. #1 also fails most gracefully — a clean no-improvement closes the genuinely-open depth axis (vs #3, which would likely regress for dt reasons and leave the architectural question unanswered).
- **Feasibility**: #1 is the lowest-effort (two hyperparameters + width arithmetic) and the easiest to make a *fair* comparison.

#1 wins on every axis that matters: strongest evidence, clearest generalization mechanism, lowest effort, best chance of a confound-free test, and it fills the one structural gap (depth) that the "capacity closed" claim asserts but never measured.

## Chosen Idea
**Selected**: Depth↔width reallocation — deeper, proportionally-narrower iso-param ResNet

**Why this idea**:
Depth is the only capacity dimension untested in 45 experiments, yet every prior "capacity closed" conclusion (EXP-004/009/038) rests exclusively on WIDTH evidence. It is also the dimension with the strongest CIFAR-specific prior — the founding ResNet paper scaled CIFAR accuracy by depth at fixed narrow width — and it is a true generalization/inductive-bias lever, exactly the class the polish-wall (project-insights) says is required for top-1 gains ("capacity or fundamentally different generalization"). Unlike the wider nets that strictly added FLOPs and underfit, a deeper-NARROWER iso-param net can be sized for ≈iso-FLOP and verified for dt-neutrality, and it has ~14 epochs of slack (EXP-007 saturation at ~77, current ~91) to absorb a modest dt rise. It is low-effort, fails gracefully (a clean no-improvement finally closes the depth axis empirically), and respects every hard constraint (train.py-only, no new deps, single 300s GPU run).

**Hypothesis**:
A deeper, proportionally-narrower ResNet at iso-param (≈4.3M; e.g. 5 blocks/stage at reduced width vs the current 3 blocks/stage at k=4), sized so measured dt stays ≈8ms and realized epochs stay ≥77 (no underfit), will generalize better per parameter than the wide-shallow k=4 baseline and raise best_test_acc by ≥0.1pp over 96.22 (i.e. ≥96.32). The null/regression outcome is that the added sequential layers raise dt enough to drop epochs below saturation (underfit) and/or that depth's per-param advantage does not materialize on this already-well-trained 32×32 net — which would close the depth axis empirically.
