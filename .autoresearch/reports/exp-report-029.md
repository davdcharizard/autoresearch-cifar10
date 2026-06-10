# Report EXP-029: SGDR — cosine annealing with warm restarts (2 cycles)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-029.md
- **Plan**: plans/plan-029.md
- **Log**: logs/exp-log-029.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) within a fixed 300s training-compute budget on a single H20, editing only `train.py`. Baseline = **96.22** (EXP-012); pass bar = **96.32** (+0.1pp). EXP-029 tests whether warm restarts (SGDR) lift top-1.

## Idea & Hypothesis
Chosen idea: **SGDR — cosine annealing with 2 warm restarts** (Loshchilov & Hutter 2017). Replace the single cosine-to-0 over the whole budget with 2 equal cosine cycles, restarting LR to PEAK at the 50%-budget mark. Selected as the largest genuinely-untried compute-neutral lever: of the LR schedule, only the PEAK was swept (settled at 0.2, EXP-016/017) and the FLOOR was closed via the SWA sweep (EXP-019/020); the schedule SHAPE / restarts were untouched. Hypothesis: the warm restart re-explores past the basin the single cosine settles into and re-anneals into a flatter/better-generalizing minimum (SGDR's single-model re-exploration benefit, distinct from the closed weight-averaging/polish class), lifting top-1 above 96.32 at unchanged ~91 epochs.

## Approach
Schedule-only change in `lr_at_fraction` (train.py L35-41) plus a new `N_CYCLES=2` constant: split [0,1] into N_CYCLES equal cosine cycles, each annealing PEAK→~0, warmup over WARMUP_FRAC of the first cycle only, restarts jumping straight to PEAK. Smoke test verified the exact shape (warmup→0.2 by frac 0.025, →0 at 0.499, restart→0.2 at 0.5, →0 at 1.0) and params unchanged (4,299,866). N_CYCLES=1 exactly reproduces the baseline schedule. No deviations from plan.

## Execution
Single run, no retries. `CUDA_VISIBLE_DEVICES=0 uv run train.py` on GPU 0, exited 0 in 403.4s. Clean compile, no NaN/Traceback. The run.log lr/loss trace confirmed SGDR fired exactly as designed: lr annealed to ~0.0000 by 49.8% budget, then JUMPED to 0.2000 at 50.1% (the restart) with the expected transient loss bump 0.87→1.16, then cycle 2 re-annealed. dt held at 8ms throughout.

## Results
- **Primary metric**: 95.55% (baseline: 96.22, delta: **−0.67pp**, −0.70%)
- **Observations**: num_epochs 91, num_steps 35292 (= baseline ~91/~35500), dt 8ms (664/705 sampled lines), peak_vram 453.8 MB. **final_test_loss 0.2076 — worse than baseline 0.195.** This is the cleanest possible test: a schedule-only change with IDENTICAL epoch count and dt to baseline, so the regression is 100% attributable to the schedule with zero throughput confound.
- **Analysis**: Hypothesis falsified, cleanly. At the 300s budget the warm restart HURTS: cycle 1 fully converged (train loss ~0.84 at 49.8%), the restart destroyed that minimum (loss spiked to 1.16), and cycle 2 (~45 epochs) re-converged to a WORSE point than the single full-budget cosine would have reached. The mechanism that helps SGDR on long-schedule ImageNet/CIFAR runs (re-exploration + snapshot ensembling over many epochs) does not pay off when the budget is short — splitting ~91 epochs into 2×~45 just gives two under-resolved anneals, and the single-model (no-snapshot-ensemble) benefit is the weak half of SGDR anyway (the ensemble half overlaps the CLOSED weight-averaging axis). This decisively closes the LR-schedule SHAPE sub-axis. Combined with the settled PEAK (EXP-016/017) and the closed FLOOR (EXP-019/020), the ENTIRE LR-schedule axis is now closed: the tuned single cosine-to-0 (peak 0.2, 5% warmup) is optimal.
- **Key Learning**: At a short fixed budget, a single full-budget cosine-to-0 beats SGDR warm restarts — splitting the budget under-converges each cycle (−0.67pp, loss worse); the LR-schedule axis (peak+floor+shape) is fully closed.

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.32) FAILED — 95.55 < 96.32; Cond 2/3 recorded informationally and PASSED (clean completion 403.4s < 600, train.py-only, params 4,299,866 unchanged, 91 evals for 91 epochs, no new deps, seed 42).
- **Review Notes**: Results fully trustworthy — the cleanest fair test of the recent batch (perfectly throughput-neutral: 91 ep / 8ms, identical to baseline; restart verified firing at 50.1%). No integrity concerns (single-forward eval unchanged, schedule is the allowed intervention class).
- **Verdict**: no-improvement
- **Verdict Basis**: verification condition failure (primary metric below bar); valid, perfectly-controlled negative.

## Unexplored Avenues
- **More cycles (3-5) or T_mult>1 (lengthening cycles)**: would only split the budget further → even more under-convergence per cycle. The 2-cycle result (already worse than 1-cycle) shows the monotone direction is toward FEWER cycles; 1 cycle (the baseline) is the optimum. Not worth testing.
- **Snapshot ensembling the SGDR cycle-end weights**: would add the ensemble benefit, but that IS weight/output averaging = the CLOSED axis (EXP-006/019/020, High, count 3). Would not exceed a tuned cosine-to-0 on top-1.
- The LR-schedule axis is **exhausted from all three angles** (peak, floor, shape) — do not revisit.

## Next Steps
- **Accept the plateau as the honest scientific conclusion** (high confidence): 30 experiments, ~22 axes now closed — all scalar knobs, the full aug family, capacity/epoch-wall, batch, weight-averaging/polish, downsampling (both sides), activation (both recipes), and now the entire LR schedule (peak+floor+shape). 96.22 is firmly the k=4 ResNet-20 / 300s / H20 ceiling, generalization-bound at fixed capacity. Per NEVER-STOP the loop continues, but expectations are calibrated to closing residual minor axes, not clearing +0.1pp.
- **Per-channel input std-norm** (low confidence, now correctly assessed as likely-REGRESSION): the frozen eval at std=(1,1,1) means changing only the train std creates a train/test BN-scale mismatch → predictable regression, not a clean null. Low informational value; only worth running to formally stamp the input-norm axis closed.
- **A genuinely throughput-POSITIVE structural idea** (low confidence): the only class that could help is one that LOWERS dt to buy epochs for the possibly-epoch-hungry TA recipe — but the net is at the ~8ms launch/compute floor and capacity is bracketed, so no such lever has surfaced. Would require a fundamentally cheaper-yet-equally-expressive block.

## Exit Action Results
- None defined.
