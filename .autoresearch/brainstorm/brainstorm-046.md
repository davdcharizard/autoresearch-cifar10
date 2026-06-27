# Brainstorm EXP-046
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external fetches; sources from model knowledge + existing base, recorded for downstream re-reading:

- **Bag of Tricks** (knowledge/papers/bag-of-tricks-zero-gamma.md, arXiv 1812.01187): besides zero-γ (failed here, EXP-018), introduces **ResNet-D** — replacing the stride-2 1×1 downsample with avg-pool + 1×1 — worth +0.5–1.0 on ImageNet at fixed epochs. The relevant principle: the strided downsample path DISCARDS 75% of spatial samples; averaging before striding preserves the information. Our CIFAR pad shortcut `x[:, :, ::2, ::2]` has exactly this defect.
- **Making Convolutional Networks Shift-Invariant Again** (Zhang, ICML 2019, arXiv 1904.11486 — model knowledge): anti-aliased downsampling (blur before stride) improves both accuracy (+0.5–1.0 CIFAR/ImageNet ResNets, fixed epochs) and shift-consistency; the aliasing argument applies to ALL strided ops, shortcuts included.
- **Standing caveat from the project record**: external fixed-epoch architecture evidence is 0-for-13 transferring here (project-insights § High, deferral entry). What distinguishes this loop's lead from the 13 failures is that it is free in EVERY measured currency simultaneously (zero params, zero init-learning, ~zero dt pending gate, zero noise change) — none of the 13 was.

## Experimental History Review

State after 46 indexed experiments: baseline 96.71 @ 1990397, bar ≥ 96.81; mean ≈ 96.57, σ ≈ 0.16; 39 consecutive non-improvements/invalids. The frontier after EXP-044/045:

- **The kernel lattice is fully charted**: fast widths = powers of two {64, 128, 256}; off-lattice = flat ~33ms tier; >256 = flat 54ms tier; grouped ~2.5–3× dense. Capacity closed in every currency INCLUDING instrument availability (exp-report-045). Do not re-derive width/depth/kernel-family candidates.
- **Every in-paradigm mechanism class is measured-closed** (recipe constants audit-complete; schedule, optimizer, batch, noise, init, activations, projection shortcuts, head pooling, attention, SAM, regularizer doses, data order, tails, resolution, BN constants, weight averaging, ensembles). exp-report-045 § Next Steps directs this brainstorm to "open the radical-structural frontier under the full screen stack: fast-lattice shapes only, deferral-free, plateau-LEVEL mechanism, in-regime evidence."
- **The screen stack every candidate must pass**: deferral (free in early heat AND epochs AND per-step gradient quality); numerics equivalence; max-statistic (plateau LEVEL only); gradient-noise neutrality; absorption (no regularizer imports calibrated to light aug); epoch-boundary; lattice shapes; effect size ≥ +0.3 plausible.
- **Unmeasured classes remaining (constructed, not catalogued)**: (a) downsample/anti-aliasing quality of the SHORTCUT path — the pad shortcut's `[::2, ::2]` strided slice throws away 75% of the identity signal at both stage transitions and aliases the rest; never touched in 46 experiments (EXP-020 changed the shortcut to learned projections — a different defect, failed on early heat + dt; the slice itself was never fixed); (b) conv-path anti-aliasing (blurpool) — same principle, but requires depthwise blur kernels = grouped family, measured 2.5–3× slow; (c) multi-point supervision (aux heads) — objective shaping beyond LS/mixup, never dosed.
- Protocol carry-overs: D0-median dt gate, dual launch gates, replicate pair for mid-band reads, ≥200-step windows, off-rung thresholds.

## Candidate Ideas

### 1. Anti-aliased shortcut: avg-pool the identity path at stage transitions (zero-param, dt-gated)
**Summary**: In `BasicBlock.forward`, replace the downsampling slice `shortcut[:, :, ::self.stride, ::self.stride]` with `F.avg_pool2d(shortcut, self.stride)` when stride > 1 (channel zero-padding unchanged). Two sites in the whole net (layer2[0], layer3[0]). Zero parameters, zero learnable state, identical tensor shapes, identical everything else. Gate at 26ms D0-median (expected dt ≈ 22.5–23.0; avg_pool2d fwd+bwd on (B,64,32²) and (B,128,16²) is a fraction of a ms).

**Reasoning**: This is the construction that survives the full screen stack. The current shortcut keeps ONE of every four spatial samples and discards the rest — a lossy, aliasing downsample on the gradient highway itself. Averaging preserves all samples' information (it is the 2×2 box filter = simplest anti-aliasing). The mechanism targets function/information quality, NOT regularization — the absorption law (which killed SE/SAM/LS imports) does not obviously apply, and it is free in every currency the deferral law prices: zero params (no early-heat learning burden — exactly what EXP-020's learned projections paid), zero epochs (pending the gate; expected ≤ +0.3ms), zero noise change, zero schedule interaction. External anchors (ResNet-D avg-pool downsample +0.5–1.0; Zhang's anti-aliasing +0.5–1.0 on CIFAR ResNets) are fixed-epoch/weak-aug — the standing 0-for-13 transfer caveat applies — but no prior failure was free in ALL currencies simultaneously; each of the 13 paid an identified toll this candidate doesn't. EXP-020 also localizes the risk: it proved the pad-shortcut's zero-channels are harmless (the residual branch fills them), so the remaining defect of the 2016 shortcut is precisely the strided slice this candidate fixes.

**Sources**: knowledge/papers/bag-of-tricks-zero-gamma.md (ResNet-D section); arXiv 1904.11486 (model knowledge); reports/exp-report-020.md (shortcut arc + what it ruled out); project-insights § High (deferral law), § Medium (kernel lattice — avg_pool is a standard dense-regime op, not grouped).

**Estimated Effort**: low (one-line forward change + CPU sanity for shape/finiteness/equivalence-at-stride-1 + standard gated composite).

**Risk Assessment**: Three failure shapes: (a) avg_pool2d misprices under compile/channels_last → gate kills in ~90s (record, invalid — low probability, it is a stock kernel); (b) the published gain is regularization-flavored after all and absorbs to a precise null at family signatures → noise-band read, downsample-quality class closed (most likely outcome by base rates); (c) the smoother shortcut changes early optimization slightly — but with identity still dominant and no learnable parts, any effect is in the function, not the optimization path. Worst case: graceful no-improvement at ~139 epochs.

### 2. Full anti-aliasing: blurpool on the stride-2 convs (conv-path, gate-first)
**Summary**: The bigger dose of the same mechanism — replace each stride-2 conv with conv(stride 1) + depthwise 3×3 blur(stride 2) (Zhang 2019), plus the Idea-1 shortcut fix.

**Reasoning (and why it is not the lead)**: Anti-aliases the main computation path, where most of the signal flows; published gains are larger than shortcut-only. But it fails the hardware screen twice: depthwise blur = grouped kernel family (measured 2.5–3× dense, EXP-042) AND stride-1 full-res convs at stage transitions roughly double those convs' FLOPs. Even optimistically this prices ≥ +3–5ms → ≥ −0.3 deficit before any gain, and the gate would likely kill it. Only worth considering if Idea 1 shows a positive slope that justifies paying for the bigger dose.

**Sources**: arXiv 1904.11486; project-insights kernel-lattice entry (grouped penalty); exp-report-042.md.

**Estimated Effort**: medium.

**Risk Assessment**: High gate-kill probability; if it ran, the deficit arithmetic already needs ≥ +0.7 true. Dominated as a first probe.

### 3. Auxiliary classifier head on stage 2 (deep supervision) — documented rejection
**Summary**: A second avg-pool+fc head on layer2's output, weighted CE added to the loss during training only; eval-mode forward returns main logits only (eval contract intact).

**Reasoning (and why rejected)**: The only unmeasured objective-shaping construction. Fails three screens: the aux head is Kaiming-init and must be LEARNED during peak heat (the EXP-018/020 deferral signature); its regularization/conditioning role is exactly what heavy aug absorbs (SE/SAM/LS precedent — three exact-deficit nulls); and its in-regime evidence is pre-BN-era (DSN 2015) or vanishing-gradient-motivated (GoogLeNet) — depth-20 BN ResNets have healthy gradients, removing the mechanism entirely.

**Sources**: model knowledge (Lee et al. DSN, arXiv 1409.5185; GoogLeNet); goal-learnings § Failed Approaches (structural arc, absorption entries).

**Estimated Effort**: low-medium.

**Risk Assessment**: Expected mean − small-deficit null or worse; no class-closing value beyond what absorption precedents already establish. Rejected.

## Idea Evaluation

- **Evidence strength**: Idea 1 has two independent published anchors (ResNet-D, anti-aliased CNNs) for the same physical mechanism, plus an in-project localization argument (EXP-020 proved the pad shortcut's OTHER defect harmless, isolating the slice). Ideas 2/3 fail hardware and absorption/deferral screens respectively before evidence even matters.
- **Mechanism clarity**: Idea 1's is the sharpest constructible: the identity path currently destroys 75% of its signal at each transition via aliasing; averaging is information-preserving at zero cost. It is a function-quality change, the category project-insights says remaining candidates must come from ("architecture free in all four currencies").
- **Expected impact**: needs ≥ +0.3 true; published dose is +0.5–1.0 fixed-epoch for fuller versions of the mechanism — shortcut-only is a fraction of that, and heavy-aug absorption may shrink it further. Honest prior: low-medium — but it is the ONLY remaining candidate with a plausible route to +0.3, and either outcome closes the downsample-quality class.
- **Risk profile**: best available — zero params, two-site one-line change, stock kernel, gate-protected, graceful failure at full epochs.
- **Feasibility**: trivial; reuses the whole validated launcher/sanity stack.

Idea 1 dominates. Ideas 2/3 are recorded to pre-empt re-derivation.

## Chosen Idea
**Selected**: Idea 1 — Anti-aliased shortcut (avg-pool the identity path at stage transitions)

**Why this idea**:
It is the only candidate the full screen stack lets through: a zero-parameter, zero-dt (pending gate), noise-neutral, lattice-preserving change to function/information quality — the exact category the program's own laws point to after closing everything else. It fixes the one defect of the 2016 shortcut that EXP-020 did NOT test (the aliasing slice, not the zero channels), with two published anchors for the mechanism, and it closes the downsample-quality class with a measurement either way.

**Hypothesis**:
Replacing the shortcut's strided slice with 2×2 average pooling at the two stage transitions preserves identity-path information that the baseline discards, raising the converged plateau: best_test_acc ≥ 96.81 at baseline-identical signatures (dt ≈ 22.5–23.0ms, ~138 epochs). Pre-registered branches: (i) best ≥ 96.81 → improvement (replicate pair first if the read lands 96.70–96.80); (ii) read within ±0.15 of the mean (96.42–96.72) at family test_loss → the anti-aliasing gain is absorbed/insufficient under TA+RE; downsample-quality class closed as the 14th external-transfer failure; (iii) GATE_KILL (D0 > 26ms) → avg_pool2d misprices under compile+channels_last (unexpected; record as a kernel datum, verdict invalid).
