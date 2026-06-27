# Brainstorm EXP-047
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external fetches; sources from model knowledge + existing base, recorded for downstream re-reading:

- **Multi-scale feature aggregation for classification** (model knowledge): feeding the classifier pooled features from MULTIPLE stages is an established pattern — hypercolumns (Hariharan et al., CVPR 2015) for dense prediction; DSN/GoogLeNet route mid-level features through training-only aux heads; fast.ai's default `AdaptiveConcatPool2d` head concatenates two pooling views because it "works better than either alone" in their applied benchmarks. The shared principle: the final linear layer is the network's information bottleneck — it sees ONLY the last stage's globally pooled vector; mid-level features carry complementary scale information that is otherwise discarded at the decision layer.
- **Identity Mappings (He 2016, arXiv 1603.05027)** — re-checked and NOT re-proposed: brainstorm-037 Idea 2 and brainstorm-044 Idea 3 both screened pre-activation out using the paper's own CIFAR ablations (gains concentrate at depth 110–1001; ≈0 or negative at depth ~20). Stands.
- **Standing caveat, now in its strongest form (exp-report-046)**: external transfer is 0-for-14 INCLUDING the toll-free case — cost-freedom is necessary but not sufficient; a candidate must additionally argue a mechanism the heavy-aug ensemble cannot itself supply (project-insights absorption entry, updated EXP-046).

## Experimental History Review

State after 47 indexed experiments: baseline 96.71 @ 1990397, bar ≥ 96.81; mean ≈ 96.57, σ ≈ 0.16; 40 consecutive non-improvements/invalids. Post-EXP-046 frontier:

- **EXP-046 (anti-aliased shortcut) absorbed to an exact null while free in every currency** → absorption is not cost-accounting. The screen stack gains its final filter: the candidate's benefit must be something augmentation/regularization cannot emulate (information access, capacity, optimization speed — and the latter two axes are closed).
- **Closed classes** (do not re-derive): capacity in every currency incl. kernel lattice; schedule/heat/noise/batch/optimizer (Muon, momentum trades); init both directions; aug dose-response peaked both sides; tails; weight & function averaging; attention (SE); activations; BN constants both directions; projection shortcuts; downsample quality (046); head POOLING OPERATOR (EXP-030 — max-pool's argmax-routed gradients lose −0.91; that closure is about the pooling op's gradient quality, NOT about which stages feed the head).
- **Never-dosed gap identified**: decision-layer information ROUTING — every experiment so far kept the classifier's input fixed at GAP(stage3) (256-d). No experiment has changed WHAT information reaches the final linear layer. EXP-030 changed how stage-3 is pooled; EXP-037 gated channels inside stages; aux heads (brainstorm-046 Idea 3) add training-only objectives — none give the eval-time classifier access to other stages.
- Protocol carry-overs: D0-median dt gate (26ms off-rung), dual launch gates, ≥200-step windows, replicate pair for 96.70–96.80 reads, integrity pre-condition before metric reads.

## Candidate Ideas

### 1. Multi-scale decision head: fc over concat[GAP(stage2), GAP(stage3)] (zero-dt, +1,280 params)
**Summary**: In `ResNet.forward`, pool stage-2's output as well: `h2 = GAP(layer2_out)` (128-d), `h3 = GAP(layer3_out)` (256-d), classify from `cat([h2, h3])` (384-d). `fc` becomes `Linear(384, 10)`. Everything else untouched. +1,280 params (+0.03%), one extra adaptive_avg_pool2d on a (B,128,16,16) tensor + a concat — sub-0.1ms; gate at 26ms regardless.

**Reasoning**: This is the only constructible candidate that passes the FULL post-046 screen stack including the new "aug-cannot-supply" filter. Mechanism: the final linear layer is an information bottleneck — at eval time it sees only the 256-d stage-3 summary; mid-level 128-d stage-2 features (higher spatial resolution, lower abstraction) are complementary AT THE DECISION and augmentation manifestly cannot supply them (it perturbs inputs; it cannot re-route which representations reach the classifier). Cost screens: zero dt (stock dense ops; gate covers), +1,280 params learned by a LINEAR layer (convex-speed learning, no early-heat burden of the EXP-020/037 kind — the fc input distribution is BN-conditioned features), zero noise change, zero schedule interaction, smooth gradients everywhere (GAP + linear — avoids EXP-030's argmax-routed credit exactly). Deep-supervision side effect (direct gradient into stage 2) is mild and smooth. Evidence: applied-practice anchor (fast.ai concat-pool default; Kaggle CV practice) plus the hypercolumn principle — honest weak-to-moderate, no controlled in-regime number, so prior is low-medium; but every alternative left is screened-out-by-law, and either outcome closes the decision-layer routing class.

**Sources**: model knowledge (Hariharan 2015 hypercolumns; fast.ai AdaptiveConcatPool2d); reports/exp-report-030.md via goal-learnings L109-110 (max-pool head closure — what this avoids); reports/exp-report-046.md (strongest-form absorption — why information ROUTING is the surviving mechanism class); project-insights absorption + deferral entries.

**Estimated Effort**: low (4-line forward/init change + CPU sanity + standard gated composite).

**Risk Assessment**: Failure shapes: (a) GATE_KILL — very unlikely (stock GAP/concat); (b) absorbed/inert null at mean band — most likely by base rates; closes the routing class cleanly at ~139 epochs; (c) mid-level features dilute the head (fc spends capacity on weaker features) → below mean band; informative negative, graceful. No destabilizing mode.

### 2. Gradient centralization on conv weights (one-line optimizer-side projection)
**Summary**: Subtract the per-filter mean from each conv weight gradient before the SGD step (Yong et al. 2020, arXiv 2004.01461) — claimed faster optimization + small accuracy gains at zero cost.

**Reasoning (and why not the lead)**: Optimization-geometry, so it dodges the absorption argument — but it changes the update arithmetic, and the project has measured update-rule changes three times (Muon EXP-028 −0.3; compensated momentum trades EXP-023/024 both directions negative; numerics-equivalence law EXP-021): every deviation from the certified nesterov-SGD arithmetic lost. Published GC gains on CIFAR are ≤ +0.3 under weak aug — under the 0-for-14 transfer record and a hostile prior class, expected value is below Idea 1's.

**Sources**: arXiv 2004.01461 (model knowledge); goal-learnings EXP-023/024/028 entries; project-insights numerics-equivalence entry.

**Estimated Effort**: low.

**Risk Assessment**: Graceful null or small deficit; closes nothing new (optimizer axis already closed — this would be a 4th confirmation).

### 3. Stochastic depth (linear-decay survival 1→0.8) — documented rejection
**Summary**: Randomly drop residual branches during training (Huang et al. 2016), identity at eval.

**Reasoning (and why rejected)**: Regularization-flavored — the EXP-046-strengthened absorption law predicts an exact null at best (TA+RE already supply the implicit-ensemble pressure; the dose-response entry says ANY added pressure on the peaked recipe loses, cf. mixup −0.46). Also batch-level drop changes the compile graph per-step (recompile risk) and per-sample drop saves no compute. Triple-screened out.

**Sources**: arXiv 1603.09382 (model knowledge); goal-learnings dose-response entry (EXP-009 mixup); project-insights absorption entry (EXP-035/036/037/046).

**Estimated Effort**: medium.

**Risk Assessment**: Expected null-to-deficit; no class-closing value beyond existing absorption precedents. Rejected.

## Idea Evaluation

- **Evidence strength**: All three are weakly evidenced in-regime (nothing in-regime remains untested with strong evidence — that frontier is exhausted by construction). Idea 1 has applied-practice anchors and, decisively, a mechanism CATEGORY argument: it is the only candidate whose benefit is definitionally outside what augmentation can emulate. Idea 2's category (update arithmetic) is 0-for-3 in-project; Idea 3's category (added regularization pressure) is closed by the dose-response bracketing AND the absorption law.
- **Mechanism clarity**: Idea 1's is sharp: the classifier's input is an information bottleneck; routing mid-level features to it changes the eval-time function class. Not regularization, not noise, not capacity-in-the-closed-sense (+0.03% params in a linear layer).
- **Expected impact**: honest low-medium for Idea 1 (no controlled in-regime anchor); but ideas 2/3 are expected-zero-or-negative BY THE PROJECT'S OWN LAWS. Either Idea-1 outcome closes the last constructible never-dosed class.
- **Risk profile**: Idea 1 fails gracefully (full-epoch null) with a 2-minute gate against the only infra risk.
- **Feasibility**: trivial; reuses the validated launcher/sanity stack.

Idea 1 dominates. Ideas 2/3 recorded to pre-empt re-derivation.

## Chosen Idea
**Selected**: Idea 1 — Multi-scale decision head (fc over concat[GAP(stage2), GAP(stage3)])

**Why this idea**:
It is the only remaining candidate whose mechanism — what information reaches the eval-time classifier — is categorically outside what the heavy-aug ensemble can supply, the exact requirement EXP-046 added to the screen stack. It is free in every priced currency (zero dt pending gate, +1,280 linear params, zero noise/schedule interaction, smooth gradients avoiding EXP-030's failure), and it opens (and will close, either way) the decision-layer information-routing class — the last never-dosed structural category constructible under the project's laws.

**Hypothesis**:
Giving the linear classifier direct access to stage-2 features alongside stage-3 raises the converged plateau LEVEL: best_test_acc ≥ 96.81 at baseline-identical signatures (dt ≈ 22.4–23.0ms, ~138 epochs, params 4,287,306). Pre-registered branches: (i) best ≥ 96.81 → improvement (replicate pair first if the read lands 96.70–96.80); (ii) read within the mean band (96.42–96.72) at family test_loss → mid-level features add no decision-relevant information at this depth/width — decision-layer routing class closed; (iii) read ≤ 96.42 → dilution: the fc misallocates capacity to weaker features — routing class closed from below; (iv) GATE_KILL (D0 > 26ms) → unexpected mispricing of GAP/concat, verdict invalid, record kernel datum.
