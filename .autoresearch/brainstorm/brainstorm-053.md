# Brainstorm EXP-053
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external fetches; sources are the project record + previously distilled anchors:

- **reports/exp-report-048.md**: the de-overhead pair (collate-side channels_last + side-stream H2D prefetch) is numerics-IDENTICAL (byte-equal values, same kernel sequence) and measured at +0.15ms/step = +1 epoch (13,515 vs 13,428 steps), read 96.57 = recipe mean exactly at family signatures. Certified free; conversion ≈ +0.02 (real but invisible alone).
- **reports/exp-report-052.md + 046**: the anti-aliased shortcut (avg_pool2d replacing the pad shortcut's strided slice; Zhang 2019 BlurPool anchor) is zero-param/zero-dt and pooled at +0.11 over the recipe mean across 3 pristine draws (96.65/96.84/96.56) — +1.2σ_mean(n=3), unresolvable from zero, individually closed as sub-bar. The point estimate is the only positive one on the board.
- **Statistics**: σ ≈ 0.16 (EXP-027), bar = mean + 1.5σ. Replicate-pair mean protocol (EXP-052-validated): 1.6% false-positive under H0; P(mean ≥ bar | true Δ = +0.13) ≈ 17%; the pair correctly declined a bar-clearing single draw last loop.

## Experimental History Review

State after 53 indexed experiments: baseline 96.71 @ 1990397, bar ≥ 96.81, 46 consecutive non-improvements. Frontier after EXP-052:

- **Every single-mechanism class is measured-closed**: recipe constants audit complete (049); loss axis closed both directions (050/051); structural classes closed (017–020/026/030/034/037/040–047); throughput exhausted at 99.3% kernel math (048); gradient-noise optimum bracketed (011/022/023/024/035); averaging closed both kinds (011/032/043); aug dose-response peaked (003/004/009/013/015/025/033); data-order closed (031/041); BN/eval-constants closed (029/038/039); the only positive-direction single datum (046) resolved sub-bar with adequate statistics (052).
- **The only unfalsified positive-direction region** (exp-report-048 Next Steps, exp-report-052 Next Steps): compound interventions of individually-certified components. The EXP-009 precedent (mixup stacked on LS+TA+RE lost −0.46) is a SAME-AXIS compounding failure — four regularizers share one currency (augmentation pressure, measured saturated). No cross-axis compound of certified-free components has ever been run.
- **Free components with non-negative measured point estimates**: anti-aliased shortcut (+0.11 pooled, function-quality axis, zero dt/params) and de-overhead prefetch (+0.02 conversion, epochs axis, numerics-identical). Nothing else qualifies: hardswish (−0.10 deficit), SWA (0 at carved-out cost), LS 0.2 (+0.01 noise), alternating flip (negative scatter), progressive resize (+0.12 read but distribution-inheritance toll and big code surface, insight caps interior ≤ +0.1).
- Protocol carry-overs: composite gates (26ms), step ledger, replicate-pair mean decision (052), trajectory numerics criterion (048).

## Candidate Ideas

### 1. Cross-axis compound of the two certified-free components (anti-aliased shortcut + de-overhead prefetch), replicate-pair n=2 MEAN decision
**Summary**: Re-apply BOTH validated diffs in one train.py: (a) EXP-046/052's shortcut change (`[::2,::2]` → `F.avg_pool2d(x, 2)` at both stage transitions), (b) EXP-048's de-overhead pair (collate-side channels_last via `collate_fn`, side-stream `CUDAPrefetcher`, in-step `.to()` lines removed). Two byte-identical gated runs; pre-registered decision = MEAN ≥ 96.81 (max never used).

**Reasoning**: Mechanism independence is measurable, not asserted: the shortcut change is zero-throughput (D0 22.5/22.0ms, family steps in all 3 draws) and acts on function quality (aliasing removal at downsampling); the prefetch is numerics-identical (byte-equal values, sanity-proven in EXP-048) and acts only on step time (+87 steps ≈ +1 epoch). They share NO currency — heat, noise, params, normalization, augmentation pressure all untouched — so the EXP-009 same-axis precedent does not apply; additivity is the default expectation for non-interacting changes. Combined point estimate: +0.11 (shortcut, pooled n=3) + 0.02 (prefetch conversion, tight) ≈ +0.13. Decision honesty: replicate-pair mean at the same 96.81 bar (1.6% false-positive under H0). Honest pass probability ≈ P(shortcut effect real, ~0.5) × P(mean ≥ bar | Δ≈+0.13, ≈0.17) + small terms ≈ 8–10% — low, but the highest-priced honest bet on the board, and every branch is terminal for the "compound of frees" question at this effect size.

**Sources**: reports/exp-report-046/048/052.md; plans/plan-046.md and plan-048.md (exact diffs + validated sanity scripts); goal-learnings Protocol Findings (replicate-pair entry, EXP-052).

**Estimated Effort**: low-medium — two known diffs re-applied (both with validated CPU sanity patterns), two gated runs (~17 min wall).

**Risk Assessment**: Graceful failure modes only: components are individually pristine at family signatures, so crash/invalid risk is engineering-only (both sanity scripts exist). Interaction risk (prefetcher feeding channels_last batches into the avg_pool model) is nil — the model consumes identical tensors as in EXP-048. Branches: (i) mean ≥ 96.81 → improvement, commit both; (ii) mean ∈ [96.61, 96.80] → weak-positive; compound-of-frees closed at the resolution limit; (iii) mean ≤ 96.60 → strong negative datum (sub-additivity), class closed; (iv) gate/contention kills → infra relaunch (max 2/run).

### 2. fp16 autocast in place of bf16 (with GradScaler) — the never-probed precision direction of the numerics axis
**Summary**: `torch.autocast("cuda", dtype=torch.float16)` + `torch.cuda.amp.GradScaler` around loss/step; everything else byte-identical.

**Reasoning (and why not the lead)**: The numerics axis is measured SENSITIVE (EXP-021: different kernel arithmetic cost ~0.4pp mid-run) but has only been probed in the faster-but-different direction, never finer-mantissa-at-same-speed: fp16 carries 10 mantissa bits vs bf16's 7 (8× finer quantization of activations/gradients) at identical tensor-core throughput on H20. CIFAR speedrun lineage (hlb-CIFAR10, airbench) trains in fp16. Against it: the same EXP-021 law cuts both ways — different rounding behavior may degrade the certified trajectory; GradScaler adds skipped-step risk at peak LR 0.4; bf16's wider exponent is what makes the current recipe scaler-free; effect size needed (≥ +0.3 one-draw) has no anchor of that magnitude. A trajectory-criterion run would resolve it, but the prior is symmetric-to-negative.

**Sources**: train.py L192/229; goal-learnings EXP-021 entry; knowledge: hlb-CIFAR10/airbench train in fp16.

**Estimated Effort**: low.

**Risk Assessment**: Symmetric outcome distribution around 0 with fat negative tail (scaler skips, overflow at peak heat); fails gracefully but likely burns a loop on a coin-flip worse than Idea 1's.

### 3. Gradient clipping (global-norm 5.0) — last unprobed optimizer knob
**Summary**: `clip_grad_norm_(5.0)` before `optimizer.step()`.

**Reasoning (and why not the lead)**: Carried from brainstorm-052: mechanism conditional on instability that does not exist (zero divergences in 40+ clean runs); at 5.0 it activates on <1% of steps (sub-σ by construction); at aggressive thresholds it becomes heat reduction — closed flat-below by EXP-049. No published anchor for stable-CIFAR gains. Recorded so it is not re-derived; not run.

**Sources**: brainstorm-052 Idea 2; goal-learnings heat entries (EXP-010/014/049).

**Estimated Effort**: trivial.

**Risk Assessment**: Expensive coin-flip on noise; dominated by Idea 1.

## Idea Evaluation

- **Evidence strength**: Idea 1 composes the only two components with measured non-negative point estimates, each validated pristine (3 + 1 runs); the additivity assumption rests on measured currency-independence rather than hope. Idea 2 has an external lineage anchor but no effect-size evidence and a two-sided risk. Idea 3 is mechanism-vacuous here.
- **Mechanism clarity**: Idea 1's two mechanisms are exact and demonstrably non-interacting (function quality at two forward sites; +1 epoch of training). Idea 2's mechanism (finer rounding → better trajectory) is plausible but unsigned. Idea 3 has no active mechanism in stable training.
- **Expected impact**: Idea 1 ≈ +0.13 expected if the shortcut effect is real — the largest honest point estimate available anywhere on the board; Ideas 2–3 are unsigned coin-flips.
- **Risk profile**: Idea 1 fails gracefully into a terminal closure of the last unfalsified region; Idea 2 can actively degrade (scaler skips); Idea 3 wastes a loop.
- **Feasibility**: Idea 1 is two known re-applications with existing sanity scripts; lowest engineering risk of the three relative to information yield.

Idea 1 dominates. Idea 2 recorded as the only remaining never-probed axis (kept for a future loop if the compound region closes); Idea 3 remains documented-weak.

## Chosen Idea
**Selected**: Idea 1 — Cross-axis compound of the certified-free components (anti-aliased shortcut + de-overhead prefetch), n=2 mean-decision at 96.81

**Why this idea**:
After EXP-052 closed the last single-mechanism datum, compound interventions of individually-certified components are the only unfalsified positive-direction region (exp-report-048/052 Next Steps). This is its sharpest first test: the only two free components with non-negative measured point estimates, on provably disjoint currencies, decided by the EXP-052-validated replicate-pair mean protocol. Every branch is terminal — pass moves the baseline; weak/null closes the compound-of-frees region with adequate statistics.

**Hypothesis**:
If the shortcut's pooled +0.11 reflects a true effect and cross-axis free components compose additively (no shared currency), the compound's true effect is ≈ +0.13 and the MEAN of two fresh byte-identical runs ≥ 96.81 with probability ≈ 17% (vs 1.6% under H0). Pre-registered branches: (i) mean ≥ 96.81 → improvement, commit both diffs; (ii) mean ∈ [96.61, 96.80] → weak-positive; compound-of-frees closed at the resolution limit (resolving +0.13 vs 0 needs n≈15); (iii) mean ≤ 96.60 → sub-additive/null; compound-of-frees closed with a negative datum; (iv) GATE_KILL/contention → infra relaunch per standard screens (both components individually measured signature-clean).
