Both the candidates and their cited sources have been read in full. Here is the adversarial review.

---

## Prioritized Feedback

**1. WRN-16-4 (idea-03): the step-indexed warmup contradicts the proposal's own thesis.** The core diagnosis — which this proposal itself restates — is that step-indexed scheduling is the baseline's bug, because throughput determines where schedule phases land. Yet its warmup is `min((step+1)/(5*len(train_loader)), 1.0)` — 975 steps at batch 256. If the throughput assumption (≤15 ms/step) is wrong, warmup silently consumes an unknown fraction of the 300 s budget, exactly the failure mode the proposal claims to eliminate. Fix: index warmup by `progress` (e.g. first 5% of charged time), matching idea-01's construction.

**2. WRN-16-4: warmup×cosine multiplication means peak LR 0.20 is never reached.** `lr = 0.20 * warmup * cosine(progress)` — cosine has already decayed while warmup ramps, so the realized peak is below 0.20 and the schedule shape differs from what the text describes. Combine with a piecewise form (warmup to peak, then cosine from peak) as idea-01 does.

**3. WRN-16-4: a silent weight-decay confound.** WD moves from the baseline's 1e-4 to 5e-4 (buried in the settings table) and appears nowhere in the risk list or ablation ladder. It is a defensible WRN convention, but on a 10× larger model over a 300 s horizon it is a substantive regularization change bundled without justification. Either justify it explicitly with the WRN literature or hold WD at 1e-4 and make 5e-4 an ablation.

**4. WRN-16-4: evaluation wall-clock is unbudgeted against the 600 s kill.** Its own throughput target implies ~102 epochs → ~102 evals of a ~10× FLOPs model on 10k images, all charged to `total_seconds` but not `training_seconds`. The baseline already runs 89 evals of a tiny model inside the 10-minute envelope; this proposal has no estimate of the new eval cost. A per-epoch eval timing estimate (or a check that faster training → more epochs → more evals doesn't breach 600 s) should be added to Verification. This is the most plausible way the run gets killed and classified a failure despite healthy training.

**5. Wide-Batch ResNet-20 (idea-01): pays most of idea-03's execution risk for a fraction of its ceiling.** It carries the same novel machinery (BF16 autocast, channels-last, time-cosine, large-batch LR scaling) but keeps the post-activation, pad-shortcut ResNet skeleton at ~1.1M params. Its own target (92.0–92.8%) sits well below idea-03's (93.0–95.0%). The risk classes are nearly identical; only the capacity bet is smaller. As written it is a sound proposal — the flaw is strategic, not internal: if the BF16/channels-last/time-schedule package must be de-risked anyway, it should buy the bigger capacity jump. Its one genuinely distinct risk — batch 512 giving ~4× fewer updates with only ~15 s of warmup at peak 0.20 — is honestly acknowledged with a batch-256 fallback.

**6. CutMix+EMA (idea-02): mechanism–limiter gap on the second half of the diagnosis.** The brainstorm's diagnosis names two gaps: the stranded LR schedule *and* 99.7% of an H20 idle under a 270K-parameter model. This proposal repairs the schedule but deliberately declines the capacity lever. That matters because ResNet-20's published fully-trained ceiling (He et al., 64k steps) is ~91.25%, and the baseline already sits at 91.51% — the architecture is near its known ceiling, so the +0.4 to +0.9 estimate is close to the plausible maximum for the whole family. The upside is structurally capped in a way the other two candidates' upsides are not.

**7. CutMix+EMA: after EMA activation the online model is never evaluated again.** If EMA (decay 0.99 per 10 steps, ~2.6-epoch window) lags the sharp late-cosine improvement, the best online checkpoint is invisible to `best_test_acc`, and the once-per-epoch rule forbids evaluating both. Mitigation within the rules: alternate which model is evaluated in the last few epochs, or shorten the EMA window near the end. The proposal's BN-buffer copying handles the classic trap correctly, but this lag risk has no monitoring path.

**8. All three: the known normalization defect is dropped without discussion.** The Collected Ideas note the `(1,1,1)` std normalization; every finalist silently omits this near-free fix. BN plausibly absorbs it (as the brainstorm itself notes), so this is minor — but the winner could absorb it at near-zero cost and risk, and the eval uses the same normalization so compatibility is preserved. At minimum, the choice to exclude it should be stated.

**No fatal issues.** No candidate violates a hard constraint, games the measurement (all preserve the frozen evaluator, timing semantics, and once-per-epoch validation), or retries a known failure — there is no experiment history to conflict with.

---

## Scored Verdict

Scores are 1–10.

**Time-Aware Pre-Activation WRN-16-4 (idea-03)**
- Evidence & reasoning: **7.5** — the capacity argument is grounded in measured VRAM/throughput facts and the stochastic-depth restraint is honestly calibrated against ShakeDrop's warnings, but the step-indexed warmup, WD confound, and unbudgeted eval time are real internal inconsistencies.
- Potential impact: **9** — WRN-16-4 at ~100 effective epochs credibly reaches 94–95% in the literature; a +1.5 to +3.5 pt gain is the largest well-reasoned ceiling on the table, with a concrete WRN-16-3 throughput fallback.

**Time-Normalized Wide-Batch BF16 ResNet-20 (idea-01)**
- Evidence & reasoning: **8** — the tightest internal logic of the three: correct piecewise schedule, sqrt-scaled LR, explicit per-component ablation ladder, and honest deferral of `torch.compile`.
- Potential impact: **6** — width-2 ResNet-20 (~1.1M params) bounds the ceiling near 92.5–93%; it de-risks the same machinery idea-03 needs while claiming much less of the available upside.

**Front-Loaded CutMix + Cosine + EMA (idea-02)**
- Evidence & reasoning: **7** — the best literature grounding per component (front-loading directly supported by the critical-period paper, one-pass CutMix correctly avoiding RegMixup's cost), but EMA is self-admittedly the least-supported piece and the online/EMA switching has an unmonitored lag failure.
- Potential impact: **5** — repairs the schedule but leaves the diagnosed hardware/capacity gap untouched on an architecture already at its published ceiling; upside is structurally capped near +1 pt.

**Pick: Time-Aware Pre-Activation WRN-16-4 with Restrained Stochastic Depth (idea-03).**

It is the only candidate that attacks both diagnosed limiters — the stranded schedule and the 99.7%-idle accelerator — and its ceiling estimate is backed by well-established WRN results rather than optimism. Idea-01 wins on internal polish but is strictly dominated in expectation: it shares idea-03's novel-machinery risks (BF16, channels-last, time-cosine, batch-scaled LR) while betting on a quarter of the capacity, and idea-03 already contains idea-01's spirit as its own WRN-16-3 fallback. Idea-02 is the safe schedule-plus-regularization play, but on a model family already at its known ceiling it cannot deliver more than ~1 pt regardless of execution quality.

The pick is conditional on four refinements from the feedback above before planning: (1) convert warmup to time-indexed, (2) use a piecewise warmup-then-cosine form so peak LR is actually reached, (3) either justify or revert the 5e-4 weight decay, and (4) add an eval-wall-clock estimate to verification so the 600 s outer kill cannot silently claim a healthy run. Stochastic depth should be the first thing dropped if the run underfits, per the proposal's own ablation ladder — it is the least essential component of the package.
