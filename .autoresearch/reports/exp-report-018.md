# Report EXP-018: Zero-init residual — γ=0 in each BasicBlock's final BN (bn2)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-018.md
- **Plan**: plans/plan-018.md
- **Log**: logs/exp-log-018.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397. Specific question: does the only never-probed recipe component — initialization — hold a gain via the zero-γ residual trick (identity-at-init), the best-evidenced init intervention for our large-batch + warmup regime?

## Idea & Hypothesis

Chosen idea: zero `bn2.weight` in all nine BasicBlocks after standard Kaiming init, making every block an identity map at init (Bag of Tricks, arXiv 1812.01187; Goyal et al. 1706.02677 — both use it in large-batch + warmup training). In-project mechanism: easing the early high-LR phase should pull the convergence onset EARLIER, lengthening the converged plateau the max-over-evals metric harvests (project-insights: the plateau is the metric's currency). Hypothesis: faster early trail, identical throughput signatures, best_test_acc ≥ 96.81. Runners-up: width asymmetry 64/128/320, airbench whitening init.

## Approach

+5-line diff in `ResNet.__init__` (`init.zeros_(m.bn2.weight)` over BasicBlocks, after `self.apply(self._weights_init)`). Everything else byte-identical to baseline; `bn2.weight` sits in the no-decay group so WD does not pin the zeros. No deviations from plan.

## Execution

One run, no retries (task b4lrsii9b, launched 10:35:05 via composite launcher + inline watchdog into a verified-free GPU 0). Pristine: zero watchdog events, 0/268 windows >30ms (mean 22.3ms), 139 epochs / 13,475 steps, startup 13.2s (warm compile cache — identical graph as predicted), total 489.7s. VRAM 1613.0MB and params 4,286,026 byte-identical to baseline: perfect attribution.

## Results

- **Primary metric**: best_test_acc = 95.72% (baseline: 96.71, delta: −0.99pp, −1.02%); bar was 96.81 — the largest deficit since EXP-008
- **Observations**: The faster-onset hypothesis INVERTED outright. The early trail was dramatically SLOWER, not faster: ep1 18.37 / ep5 35.26 / ep8 55.16 / ep10 68.01 vs the same-throughput comparator EXP-017 (63.76@5, 75.06@8) — at epoch 5 the zero-γ run is ~28pp behind. It stayed 6–11pp behind through the middle (78.91@60, 89.97@100) and converged to a flat plateau (final eight evals 95.63–95.72, final ≈ best) a full point below baseline. By the EXP-008 diagnostic this is a convergence-level deficit: the run finished its schedule, it just learned less with it.
- **Analysis**: The mechanism is the project's master failure mode — deferral — entering through the one door it hadn't used yet (init). Identity-at-init means the network spends the warmup and peak-LR phase (the hottest, most valuable heat in the time-keyed schedule) as effectively a stem + linear head while the nine γs grow from zero; by the time full depth is "on", the schedule is already annealing. The literature's gains do not transfer for two stacked reasons: (1) they are measured at fixed EPOCHS on long schedules where the turn-on cost amortizes to nothing, while our wall clock makes early heat unrecoverable (project-insights Medium: fixed-iteration intuitions invert); (2) zero-γ's primary documented benefit is enabling LARGER peak LRs / bigger batches without divergence — but our peak (0.4) is already certified optimal with margin (EXP-010: 0.6 diverged in quality, not stability), so we paid the cost of extra early stability we did not need and could not spend. The init axis's cheap end is now closed with a clean, strongly-signed data point; thirteen consecutive misses. Strikingly, EXP-017+018 together sharpen what the 300s regime rewards: not easier optimization, not more parameters — only interventions that increase what is LEARNED PER UNIT OF SCHEDULE HEAT, and nothing tried since EXP-006 has done that.
- **Key Learning**: Zero-γ init inverts under fixed wall clock: identity-at-init defers effective capacity through the hottest schedule phase (ep5: 35% vs 64%) for stability headroom the certified-optimal peak LR doesn't need — init tricks priced in early heat all fail here.

## Verification

- **Conditions**: pre-condition contention sanity CLEAN (139 epochs exactly on projection; 0/268 windows >30ms); condition 1 FAILED (best_test_acc 95.72 < 96.81); conditions 2–3 skipped per first-failure stop (observed informally: 489.7s ≤ 600 rc=0; 139 evals = 139 epochs — both would have passed)
- **Review Notes**: trustworthy — metric matches the eval trail (best 95.72 @ ep 135); VRAM/params/startup byte-identical to baseline confirms the identical-graph claim, so the deficit is attributable to the init alone
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid clean run; primary-metric necessary condition not met)

## Unexplored Avenues

- **Partial zero-γ (e.g. only stage-2/3 blocks, or γ=0.5 instead of 0)**: would soften the turn-on deferral — but the dose-response sign is so strongly negative (−0.99pp at full dose) that even the zero-crossing is unlikely to sit above +0.1pp. Low interest.
- **Whitening first-conv init (airbench)**: unlike zero-γ it ADDS information at init (decorrelated features from step 0) rather than removing expressivity — the deferral objection does not apply; but implementation is medium-high effort, our normalization (std=1) differs from airbench's, and TA/RE distort patch statistics. The one init-axis direction not closed by this result. Moderate-low interest.
- **Init-scale tuning (e.g. Kaiming fan-out vs fan-in, or scaled residual branches à la Fixup)**: same family; any variant whose mechanism is "more early stability" is now predicted to lose by the EXP-018 sign; only variants whose mechanism is "more early LEARNING" merit a try.

## Next Steps

1. **Width asymmetry at constant alignment (64/128/320, [3,3,3])** — the last untried capacity move; preserves early-stage depth (the EXP-017 failure isolate) and adds stage-3 params at ~+17% FLOPs (~125–130 epochs, must measure compiled dt). Confidence: low.
2. **Learning-rate floor / minimum-LR tail within cosine (e.g. anneal to 0.005 instead of ~0)** — would keep late evals refining and might raise the plateau level; CAUTION: flagged risk of being a variance-increasing max-statistic exploit (goal-learnings EXP-011 insight) — only valid if it raises the plateau MEAN, which must be argued in brainstorm before trying. Confidence: low.
3. **Accept-the-optimum remains a non-experiment** (no-seed-hacking constraint); the EXP-017/018 pair strengthens the case that 96.71 is the measured optimum of this recipe family: neither free capacity nor easier optimization moves the converged plateau. Confidence: n/a (framing note).

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
