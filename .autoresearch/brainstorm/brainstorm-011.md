# Brainstorm EXP-011
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new search this loop — the decision rests on in-project measurements plus standard, widely-replicated practice:
- **EMA weight averaging** (standard practice: timm recipes, PyTorch ImageNet references, Polyak averaging literature): evaluating an exponential moving average of weights consistently matches or beats raw snapshots; gains shrink but persist under cosine-annealed schedules. No in-project datapoint yet — the last untried orthogonal lever.
- **Fixed-time deferral principle** (project-insights, EXP-010): levers must pay off without extending the schedule. EMA qualifies — it harvests existing noise rather than deferring progress.
- **Dead-end identified by first principles this loop**: GPU-resident data / loader optimization buys NOTHING for the metric — the per-step fetch happens before the dt timer starts, so loader stalls consume wall clock (480.8s vs the 600s cap = ample headroom) but zero training budget. Epochs are purely GPU-time-bound. This removes exp-report-010's Next-Step #3 from consideration permanently.

## Experimental History Review

- **Trajectory**: 91.97 → 93.16 → 95.23 → 94.41(f) → 96.06 → 96.23 → 95.12(f) → 96.71 → 96.00(f) → 95.76(f) → 96.25(f) → 96.14(f). Baseline: **96.71 @ 1990397** — unbeaten for five loops.
- **Closed axes**: capacity (bidirectionally: starvation High count-3 + ResNet-14 convergence), regularization (dose-response crossed zero at mixup), peak LR (bracketed from above; EXP-006's healthy convergence shape argues 0.4 is near-optimal). All four failures since EXP-006 share the fixed-time deferral mechanism (project-insights Medium).
- **What the baseline's residual structure offers**: (a) final-epoch eval noise of ±0.1pp epoch-to-epoch at convergence (EXP-009/010 tails) — harvestable by averaging; (b) WARMUP_FRAC 0.15 spends ~45s ramping up — possibly compressible; (c) final≈best with a small gap (96.65/96.71 at EXP-006) — the run converges properly, little left from schedule tweaks.
- **Untried gaps**: EMA eval (never tried), schedule-shape constants (WARMUP_FRAC), PEAK_LR downward probe (0.3), augmentation fade-out near schedule end, compile max-autotune (projected sub-bar).

## Candidate Ideas

### 1. EMA weight averaging for evaluation (decay 0.995)
**Summary**: Maintain an eager EMA copy of the model (`copy.deepcopy(base_model)`, grads off). After each `optimizer.step()`: `torch._foreach_lerp_` all EMA params toward live params (decay 0.995 ≈ 1.4-epoch half-life at 97 steps/epoch) and copy BN buffers. Evaluate the EMA copy once per epoch (replacing the raw eval — the once-per-epoch rule allows only one).

**Reasoning**: The only widely-validated lever not yet tried, and it respects the fixed-time principle: instead of deferring progress, it harvests variance already being paid for — the observed ±0.1pp epoch-to-epoch eval noise at convergence is exactly the bar's size. Late in training (LR→0) the EMA converges to the raw weights from above: it is approximately an average over the trailing ~1.5 epochs of snapshots, which dominates a single snapshot in expectation under noise. The update is a fused foreach lerp over ~65 tensors (~0.3–0.5ms/step ≈ 2–3 epochs of cost — the main risk). All EMA machinery stays out of the compiled graph; eval still goes through the frozen `Eval` on the eager EMA model.

**Sources**: timm/PyTorch ImageNet recipes (standard practice); EXP-009/010 final-epoch tails (run.log noise structure recorded in exp-logs); project-insights § fixed-time deferral.

**Estimated Effort**: low-medium (~12 lines: deepcopy, per-step foreach lerp + buffer copy, eval-target switch)

**Risk Assessment**: Expected +0.0–0.3pp vs a −0.05pp throughput tax — genuinely uncertain, but the bar is only +0.1. Failure clean and informative (closes the averaging direction). Bug surface: must deepcopy BEFORE `torch.compile` wraps the model; must eval the EMA copy, not the live one; buffers must track (BN stats), else eval is garbage — an epoch-1 sanity check catches this.

### 2. WARMUP_FRAC 0.15 → 0.08 (redistribute budget from warmup to anneal)
**Summary**: Halve the warmup share of the one-cycle, giving the cosine descent ~7% more of the 300s budget.

**Reasoning**: EXP-010 showed recovery time is the scarce resource; warmup is the schedule's least-productive phase for test accuracy. But the absolute effect is small (the freed ~20s lands mostly in the mid-schedule plateau, not the productive tail), and short warmups at batch 512 risk early instability that BN usually — but not always — absorbs.

**Sources**: arXiv 1708.07120 (warmup length secondary to peak/anneal); exp-report-010 § Unexplored Avenues.

**Estimated Effort**: trivial (one constant)

**Risk Assessment**: Likely within noise either way (±0.15pp); failure clean; lowest information content of the three (a null tells us little).

### 3. Augmentation fade-out: drop TA+RE for the final ~15% of the budget
**Summary**: At progress ≥ 0.85, swap the train dataset's transform to crop+flip only (recreate the DataLoader once — persistent workers cache the old transform), letting the low-LR tail fit cleaner data.

**Reasoning**: Mechanistically aligned with the fixed-time principle — the final epochs make the most metric progress, and heavy augmentation deliberately corrupts exactly those epochs' gradients. Related tricks (de-randomized/reduced augmentation late) appear in speedrun recipes. But: the loader swap costs a few seconds of wall clock and worker warm-up, mid-experiment loader recreation is new bug surface, and the closest published analogue is weaker evidence than EMA's ubiquity. Risk of overfitting the tail (train/test gap reopens precisely when LR can't repair it).

**Sources**: airbench (2404.00498) reduced-randomness augmentation; fixed-time deferral insight; goal-learnings § saturation entry (does NOT conflict: this removes augmentation rather than adding).

**Estimated Effort**: medium (transform swap + one-time loader recreation keyed on progress)

**Risk Assessment**: Expected −0.2 to +0.3pp — wide; loader-recreation bug surface; failure semi-ambiguous (overfit vs implementation).

## Idea Evaluation

**Evidence strength**: Idea 1 is standard practice across modern training recipes with consistent small positive effects, including under annealed schedules; Ideas 2–3 rest on thinner, more situational evidence. Within this project, Idea 1 is also the only candidate whose target signal (±0.1pp terminal eval noise) has been directly measured in the last two exp-logs.

**Mechanism clarity**: Idea 1 — variance reduction over trailing snapshots; pays immediately, defers nothing. Idea 2 — budget redistribution with diffuse payoff. Idea 3 — clean-data fine-tuning of the tail; plausible but double-edged (gap reopening).

**Expected impact**: Idea 1: +0.0–0.3pp net of the ~2–3-epoch update tax. Idea 2: ±0.15pp. Idea 3: −0.2–+0.3pp.

**Risk profile**: Idea 1's failure modes are either a clean null or an implementation bug caught at epoch 1 (garbage BN buffers → near-random eval). Idea 3 carries mid-run loader surgery. Idea 2 is safe but low-information.

**Feasibility**: Ideas 1–2 small; Idea 3 medium. Idea 1 dominates on evidence and measured-signal alignment.

## Chosen Idea
**Selected**: Idea 1 — EMA weight averaging for evaluation (decay 0.995)

**Why this idea**:
The last untried lever with broad external validation; it harvests variance the runs already pay for (the measured ±0.1pp terminal eval noise — the same size as the pass bar) instead of deferring progress, which is the mechanism that killed the last four experiments; and its failure modes are clean (null) or immediately detectable (epoch-1 garbage eval).

**Hypothesis**:
Evaluating an EMA (decay 0.995, per-step foreach update with BN-buffer tracking) of the otherwise-frozen compiled 4x recipe will raise best_test_acc from 96.71% to ≥96.81%, because the EMA approximates an average over ~1.5 trailing epochs of snapshots whose individual evals fluctuate ±0.1pp at convergence, and averaging dominates single snapshots under that noise; predicted execution signatures are dt ≤ 23ms (update tax ≤ 1ms), ~135–139 epochs, EMA evals LAGGING raw-run trajectories mid-schedule by a few pp (expected — the average trails a moving target), then converging to ≥ the raw trajectory in the final low-LR epochs.
