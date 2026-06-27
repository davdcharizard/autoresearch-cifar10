# Brainstorm EXP-012
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new search this loop — the knowledge base already holds the decisive references:
- **airbench (knowledge/README.md → arXiv 2404.00498)**: the current CIFAR-10 speedrun record trains at **batch 1024** on a similar-scale network — direct external precedent that batch 1024 is in the healthy regime for fast CIFAR training.
- **Linear LR scaling (Goyal et al., cited in train.py's own PEAK_LR comment)**: batch and LR scaled together preserve gradient-noise scale; the recipe's 0.4 @ 512 came from exactly this rule (0.1 × 512/128). Scaling to 0.8 @ 1024 is the same principle, not a new-LR experiment.
- **In-project measurement (EXP-006/exp-report-006.md)**: throughput is the one axis that consistently converts to accuracy — +25 epochs gave +0.48pp. Patterns High: "throughput gains keep converting."

## Experimental History Review

- **Trajectory**: 91.97 → 93.16 → 95.23 → 94.41(f) → 96.06 → 96.23 → 95.12(f) → **96.71** → 96.00(f) → 95.76(f) → 96.25(f) → 96.14(f) → 96.46(f). Baseline **96.71 @ 1990397** — six consecutive misses.
- **Closed axes**: capacity (bidirectional, High count-3), regularization (saturated, dose-response crossed zero), peak LR upward at fixed batch (EXP-010), EMA/smoothing-class eval techniques (EXP-011 — best_test_acc is a max-statistic that rewards variance; smoothing collapses the peak).
- **The one axis that always paid**: throughput → epochs (bf16 stack +3.75x, compile +1.22x → +0.48pp). Per goal-learnings Medium, the precision/layout lever is exhausted and "further throughput needs GPU-resident data or bigger batches" — GPU-resident data was eliminated by first principles in brainstorm-011 (loader stalls land outside the timed budget), leaving **bigger batches** as the last untried throughput lever. BATCH_SIZE has sat at 512 since EXP-000.
- **Untried gaps**: batch size upward (with linearly-scaled LR), WARMUP_FRAC, WD re-tune (untouched since EXP-000), PEAK_LR 0.3 downward.
- **Integrity flag (idea discarded)**: exp-report-011 floated a terminal LR floor to RAISE eval variance and fish the max-statistic. That improves best_test_acc without improving the model (it would hurt final/mean accuracy) — the benefit would not survive a change of metric composition, i.e. reward hacking per the goal's intent. Not pursued.

## Candidate Ideas

### 1. BATCH_SIZE 512 → 1024 with linearly-scaled PEAK_LR 0.8 (throughput via utilization)
**Summary**: Double the batch and the peak LR together (Goyal linear scaling, the same rule that set 0.4 @ 512). Steps/epoch halves (97 → 48); if the H20 executes a 1024-batch step in less than 2x the 512-batch dt (likely — at 22ms the 4.3M-param 32x32 kernels underutilize a 989-TFLOPS-class GPU; H20 alignment is satisfied, channel widths unchanged), img/s rises and the run fits MORE epochs in the fixed 300s. dt-gate at step ~100 makes the bet cheap: measured dt ≥ ~44ms ⇒ no throughput gain ⇒ kill per protocol.

**Reasoning**: Targets the only mechanism with a 100% hit rate in this project (throughput → epochs → accuracy). It is NOT a rerun of EXP-010: that raised LR at fixed batch (changing the noise scale); this holds the noise scale constant by scaling both. External precedent is strong (airbench trains at 1024). Eval count rises with epochs (~0.85s each) but projected total stays ≤ ~500s. VRAM roughly doubles activations (~1.6 → ~3GB) — trivial on an H20. Risks: (a) large-batch generalization gap could eat the epoch gain; (b) if dt scales ≥ linearly the run degenerates into an EXP-010-style hot-LR experiment — mitigated by the dt-gate kill; (c) torch.compile recompiles for the new shape (startup cost only, warmup absorbs it).

**Sources**: knowledge/README.md (airbench 2404.00498, batch 1024); train.py L23 comment (linear scaling rule); reports/exp-report-006.md (epochs→accuracy conversion); goal-learnings § Patterns Medium (bigger batches named as the remaining throughput lever).

**Estimated Effort**: trivial diff (two constants), moderate verification care (dt-gate, epoch projection)

**Risk Assessment**: Expected −0.3 to +0.4pp; cheap kill at step 100 if throughput doesn't materialize; clean attribution (single principled lever). The widest upside of the remaining candidates.

### 2. WARMUP_FRAC 0.15 → 0.08 (redistribute budget from warmup to anneal)
**Summary**: Halve the warmup share, giving the cosine descent ~7% more of the 300s.

**Reasoning**: Cheapest surviving schedule-shape tune; EXP-010 showed recovery time is scarce, and warmup is the least productive phase. But the freed ~20s lands mostly mid-schedule, the expected effect is within noise (±0.15pp), and a null result teaches almost nothing (low information).

**Sources**: exp-report-010/011 § Next Steps; arXiv 1708.07120 (warmup length secondary).

**Estimated Effort**: trivial (one constant)

**Risk Assessment**: Safe, low-information, likely within noise.

### 3. WEIGHT_DECAY re-tune (5e-4 → 2.5e-4) on the heavily-augmented recipe
**Summary**: Halve explicit WD — LS+TA+RE supply far more regularization than existed when 5e-4 was inherited from the 2016 recipe (never revisited since EXP-000).

**Reasoning**: EXP-009 proved the recipe sits AT the regularization optimum's edge (mixup pushed it over); explicit WD is part of the same total-regularization budget, so reducing it could relieve the same over-constraint mixup exposed — but symmetric logic says the optimum may equally be at 5e-4 already (EXP-006 converged healthily). Mechanism is plausible but direction is a coin flip, and WD interacts with the LR schedule making a single-value probe weakly informative.

**Sources**: goal-learnings § Patterns High (dose-response saturated; "PEAK_LR/WD never revisited since EXP-000"); reports/exp-report-009.md.

**Estimated Effort**: trivial (one constant)

**Risk Assessment**: Expected ±0.2pp, direction uncertain; clean failure; moderate information (maps the WD axis).

## Idea Evaluation

**Evidence strength**: Idea 1 has the strongest support stack — external (airbench at 1024), theoretical (linear scaling preserves dynamics), and in-project (throughput→accuracy converted every time it was tried). Ideas 2–3 rest on plausibility arguments with no comparable precedent.

**Mechanism clarity**: Idea 1 — better GPU utilization → more epochs in fixed time → the EXP-006-validated conversion; testable at step 100. Idea 2 — diffuse budget shift. Idea 3 — total-regularization rebalancing, direction unknown.

**Expected impact**: Idea 1 is the only candidate with upside beyond noise (+0.2–0.4pp if dt_1024 ≲ 38–40ms ⇒ ~155–165 epochs). Ideas 2–3 are ±noise probes.

**Risk profile**: Idea 1's main risk (no throughput gain) is killed at step ~100 for pennies; its research risk (large-batch gap) yields a clean, informative negative that closes the batch axis. Ideas 2–3 fail silently into the noise band.

**Feasibility**: All trivial diffs. Idea 1 needs the most careful verification but the protocol (dt-gate, epoch projection, wall-clock cap check) is already validated goal infrastructure.

## Chosen Idea
**Selected**: Idea 1 — BATCH_SIZE 1024 + PEAK_LR 0.8 (linear scaling)

**Why this idea**:
It attacks the only axis that has never failed to convert (throughput → epochs), is the last untried throughput lever after precision/compile/GPU-residency were exhausted or eliminated, has the strongest evidence stack of the remaining candidates, and its dominant failure mode is detectable at step ~100 for near-zero cost.

**Hypothesis**:
Doubling batch to 1024 with PEAK_LR 0.8 (linear scaling, preserving gradient-noise scale) will execute steps at dt < 44ms on the H20 (utilization headroom at 22ms/512), raising img/s and fitting ≥ ~150 epochs in the 300s budget, and the extra ~12+ epochs will convert to accuracy per the EXP-006 mechanism, lifting best_test_acc from 96.71% to ≥ 96.81%; predicted signatures are dt ~36–42ms at step 100 (kill if ≥ 44ms), ~48 steps/epoch, mid-schedule trajectory tracking baseline (NOT the 3pp-depressed EXP-010 shape — the noise scale is preserved), VRAM ~3GB, total wall clock ≤ ~520s.
