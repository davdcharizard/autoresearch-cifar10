# Brainstorm EXP-001
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Wide Residual Networks (Zagoruyko & Komodakis, arXiv 1605.07146)** (https://arxiv.org/abs/1605.07146 ; https://github.com/szagoruyko/wide-residual-networks)
  Widening residual blocks beats adding depth on CIFAR-10: networks with 16/22/40 layers gain consistently at 1–12× width; WRN-40-4 matches ResNet-1001 accuracy while training 8× faster. Directly supports widening the existing ResNet-20 rather than deepening it.
- **Knowledge base (saved loop-000)**: cifar10-fast and airbench (knowledge/README.md § References) — every speedrun architecture is shallow-wide, never deep-thin; channel widths 64–512 at CIFAR scale.

## Experimental History Review

- **Current best**: 93.16% (EXP-000, commit be45820) — budget-matched one-cycle recipe on unchanged ResNet-20. Index: experiment-indices/maximize-cifar10-test-accuracy.tsv.
- **Validated patterns** (goal-learnings § Patterns): time-keyed one-cycle LR is composable — keep under any architecture change; bf16+TF32+channels_last+batch 512 gives 3.75× img/s and the precision/layout lever is exhausted; host DataLoader is the remaining throughput bound.
- **Protocol constraint** (goal-learnings § Protocol Findings, High Importance): EXP-000 finished at 596.7s of the 600s cap — per-epoch eval (~0.85s × 345 epochs) nearly busted the wall clock. Any new experiment that *raises* epoch count is dangerous; experiments that lower epoch count (heavier model, bigger batch) relieve this.
- **Untried gaps**: architecture capacity (still the 270k-param 2016 model), GPU-resident data pipeline, augmentation upgrades (Cutout/RandomErasing), LR/warmup micro-tuning.

## Candidate Ideas

### 1. Widen ResNet-20 4× (WRN-style) on the validated recipe
**Summary**: Change only the channel widths: (16, 32, 64) → (64, 128, 256) — i.e., `ResNet(num_blocks=3, width_mult=4)`, ~4.3M params (vs 270k). Keep everything from EXP-000: time-keyed one-cycle (peak 0.4, 15% warmup), bf16/TF32/channels_last, batch 512 nesterov, selective WD 5e-4, label smoothing 0.1, eval once per epoch. The zero-pad shortcut logic already handles arbitrary widths.

**Reasoning**: EXP-000 removed the recipe bottleneck; capacity is now binding (270k params is tiny for 300s of H20 compute). WRN evidence is direct: width gains are consistent at 16-layer depth up to 12×, and wide trains faster per unit accuracy than deep. Crucially this *relieves* the protocol risk: ~16× FLOPs/image cuts throughput to roughly 10–20k img/s → ~60–120 epochs → eval overhead drops from ~295s to ~60–100s, pulling total_seconds comfortably under the cap. The validated time-keyed schedule automatically adapts to the new throughput — no retuning needed for schedule horizon.

**Sources**: arXiv 1605.07146 (WRN); goal-learnings § Patterns (recipe composability), § Protocol Findings (eval-overhead cap); reports/exp-report-000.md § Next Steps.

**Estimated Effort**: low (a width multiplier in the model constructor + channel constants; recipe untouched)

**Risk Assessment**: Peak LR 0.4 was tuned for the thin net; wider nets sometimes prefer slightly lower peak — one-cycle warmup mitigates divergence risk. Fewer total steps (~60–120 epochs × 97 steps) could undertrain the larger model; WRN results at ~similar effective epochs suggest 16-layer-wide converges fast. Worst case: accuracy below 93.26 bar → no-improvement, discard.

### 2. ResNet-9 shallow-wide architecture swap (cifar10-fast/airbench style)
**Summary**: Replace the model with Page's ResNet-9 (conv-pool prep 64, layers 128/256/512, two residual blocks, maxpool head with scaled logits) under the EXP-000 recipe.

**Reasoning**: The proven speedrun architecture class — 94% in 26–79s on older GPUs. Higher ceiling per wall-clock second than any ResNet-20 variant.

**Sources**: knowledge/README.md § References (cifar10-fast, airbench).

**Estimated Effort**: medium (full model rewrite; head/pooling differences; recipe interactions — Page's net uses different LR scale, no BN-bias decay, logit scaling 0.125)

**Risk Assessment**: Bigger structural change in one loop; recipe (peak LR, WD) was validated on the BasicBlock topology and may transfer poorly; attribution muddier. A failed run wastes the loop where Idea 1 is near-surely positive.

### 3. Cutout/RandomErasing augmentation on the current net
**Summary**: Add `transforms.RandomErasing` (torchvision, already a dependency) after normalization in the train transform, p=0.5, scale (0.02, 0.25) — Cutout-equivalent regularization.

**Reasoning**: DeVries & Taylor report ~+0.5–1pp on CIFAR-10; cheap and orthogonal. But regularization pays off most when capacity is high relative to data — the current 270k-param net at 345 epochs showed no overfitting signature (final ≈ best), so the gain now is likely at the low end.

**Sources**: torchvision docs (RandomErasing); arXiv 1708.04552 (Cutout).

**Estimated Effort**: low (one transform line)

**Risk Assessment**: May slightly *hurt* at current capacity/epoch count if the model is underfitting; better sequenced after the capacity bump where it has more to regularize.

## Idea Evaluation

**Evidence strength**: Idea 1 has the strongest directly-applicable evidence — WRN's systematic width study at exactly this depth (16-layer class) plus every speedrun using wide-shallow nets, composed with our own validated recipe (EXP-000). Idea 2's evidence is equally strong for the architecture class but weaker for *recipe transfer* (Page's net comes with its own tuned recipe we'd be partially discarding). Idea 3 has good literature but a weak match to the current regime (no overfitting signature yet).

**Mechanism clarity**: Idea 1 — capacity is the binding constraint now that the recipe is fixed; more channels per layer = more accuracy per step, and the time-keyed schedule self-adapts to the lower throughput. Clean single-variable change. Idea 2 — same capacity mechanism but entangled with topology and recipe changes. Idea 3 — regularization mechanism mismatched to an underfitting regime.

**Expected impact**: Idea 1: +0.6–1.5pp expected (WRN-16-4-class accuracy at moderate epochs is ~94.5–95% with full training; 60–120 one-cycle epochs should capture most of it). Idea 2: similar or higher ceiling, lower landing probability this loop. Idea 3: +0.2–0.5pp at best now.

**Risk profile**: Idea 1 fails gracefully and *reduces* the wall-clock-cap risk flagged in goal-learnings (fewer epochs → less eval overhead). Idea 2 risks recipe mismatch. Idea 3 risks a marginal/no gain that burns a loop.

**Feasibility**: Idea 1 is a ~5-line diff. Idea 2 is a rewrite. Idea 3 is 1 line but low-impact.

## Chosen Idea
**Selected**: Idea 1 — Widen ResNet-20 4× (WRN-style) on the validated recipe

**Why this idea**:
Single-variable capacity increase with the strongest setting-specific evidence (WRN width study at 16-layer depth), composed on top of the validated EXP-000 recipe, trivially implementable, gracefully failing, and it actively relieves the 600s wall-clock risk by cutting epoch count ~3–5×.

**Hypothesis**:
Widening channels (16,32,64) → (64,128,256) (~4.3M params) under the unchanged EXP-000 recipe will raise best_test_acc from 93.16% to ≥93.8%, because model capacity — not the training recipe — is now the binding constraint, and WRN evidence shows 4× width at this depth yields >1pp gains; secondarily, total_seconds will drop below ~500s as epoch count falls to ~60–120.
