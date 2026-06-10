# Report EXP-025: Large-batch throughput exploitation (batch 256 + linear LR scaling)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-025.md
- **Plan**: plans/plan-025.md
- **Log**: logs/exp-log-025.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within a fixed 300s training budget on one H20. Baseline = **96.22%** (EXP-012, commit 6c417a4); bar = **96.32%** (+0.1pp).

## Idea & Hypothesis
Chosen idea: exploit the launch-bound regime to buy more effective epochs by raising `BATCH_SIZE` 128→256 with linear LR scaling (`PEAK_LR` 0.2→0.4, Goyal 2017) and a longer warmup (0.05→0.08). The enabling premise was a code fact: the 300s budget gates on `total_training_time` = Σ(per-step compute `dt`), with the timer starting AFTER the dataloader yields (train.py L218-242). So #steps ≈ 300/mean(dt) and effective epochs = steps·batch/50000. IF per-step `dt` stays ~flat as batch grows (launch-bound, ~8ms/step, VRAM ~0.5GB/98GB), a larger batch processes ~2× images per second of compute ⇒ ~2× effective epochs for free — the only known lever that ADDS training without hitting the epoch wall. Hypothesis: at batch 256, dt stays ~8-9ms, epochs rise above 91, and (because the strong-TA recipe is not epoch-saturated like the pre-TA EXP-007) accuracy clears 96.32. Explicitly falsifiable: if dt ~doubles (compute-bound) the premise is wrong.

## Approach
Three hyperparameter edits in `train.py`: `BATCH_SIZE` 128→256 (L22), `PEAK_LR` 0.2→0.4 (L23), `WARMUP_FRAC` 0.05→0.08 (L24). Added two core-torch `DataLoader` kwargs (`persistent_workers=True`, `prefetch_factor=4`) purely as a wall-clock safeguard — the accuracy mechanism is compute-`dt`-gated and independent of dataloader speed. No architecture change → params expected unchanged. The time-fraction cosine schedule consumes PEAK_LR directly, so it auto-anneals over the 300s regardless of batch. Scope = train.py only; no new deps. No deviations from plan.

## Execution
Single run, no retries. Clean startup (1.9s), clean compile, no NaN, no Traceback, total_seconds 377.5 < 600. Completed **72 epochs** / **13,964 steps** in 300s of compute, peak VRAM 870MB.

## Results
- **Primary metric**: best_test_acc = **93.84%** (baseline: 96.22, delta: **−2.38pp**, −2.47%) — the largest regression in the project to date.
- **Observations**: **Launch-bound premise FALSIFIED.** Per-step `dt` rose from baseline ~8ms (batch 128) to **~15ms steady-state and ~24-28ms during early/warmup steps** (run.log early `dt: 26/28/24ms`, late `dt: 15ms`); mean dt ≈ 300s/13964 ≈ 21.5ms ≈ 2.7× baseline. So the net is **compute-bound at batch 256**, not launch-bound. img/s only reached ~17,500 late-run (+12% over baseline ~15,600) with early samples ~9,700 — the launch-overhead amortization is small and partial, nowhere near the 2× needed to hold epochs flat.
- **Analysis**: Because dt more than doubled, batch-256 processed FEWER images AND collapsed optimizer **updates 61%** (baseline ~35,490 steps → 13,964). Far fewer gradient updates plus the 2× peak LR (0.4) drove the cosine schedule to a much worse optimum: final_test_loss ROSE 0.195→0.258 (under-resolved optimization, not overfit). This is the mirror image of the compute-ADDING failures (EXP-004/009/015/024): there, added FLOPs cut epochs; here, the larger batch raised per-step cost AND halved steps/epoch, cutting BOTH images and updates. The result cleanly resolves a long-standing assumption: k=4 is launch-bound ONLY at batch 128 (8ms); at batch 256 it is compute-bound (dt scales ~with batch), so batch scaling cannot buy free epochs — it strictly reduces updates and regresses. The large-batch generalization gap (Smith 2018 predicts EQUAL accuracy at equal epochs, never better) compounds the update collapse.
- **Key Learning**: At batch 256 the k=4 net is compute-bound (dt 8→15-26ms), so larger batches do NOT yield free epochs under the compute-gated budget — updates collapsed 61% and accuracy fell −2.38pp; the launch-bound headroom exists only at batch 128.

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.32) FAILED (93.84); Conds 2–3 skipped per protocol (would pass — clean 377.5s run, train.py-only, params 4,299,866, eval-count 72 == epochs, no new deps, seed intact).
- **Review Notes**: Trustworthy as a measurement — clean run, scope/params intact, intervention is within the explicitly-allowed class (batch size is fair game per the goal statement), no integrity issue. The regression is real and mechanistically explained (compute-bound → fewer updates), NOT a confound to be discounted: it is a clean negative test of the launch-bound premise.
- **Verdict**: no-improvement
- **Verdict Basis**: necessary condition failure (primary metric below bar). Valid result, no constraint violation.

## Unexplored Avenues
- **Smaller batch increase (160 or 192) with sqrt or partial LR scaling**: at batch 256 dt already ~doubled, so any batch >128 likely leaves the launch-bound regime; a smaller bump might keep dt closer to flat but would yield at most a marginal epoch gain and still risks the update/generalization trade-off. Low value — the dt-vs-batch curve shows the launch-bound headroom is essentially consumed by batch 128.
- **Batch 256 with LR held at 0.2 (no scaling)**: would isolate whether the regression is the LR-0.4 overshoot vs the update collapse. But updates still collapse 61% regardless of LR, so it would still regress; only diagnostic value, low expected gain.
- **The deeper truth**: the compute-gated budget rewards MORE updates at SMALLER effective batch, not fewer-larger. Going the OTHER direction (batch <128) would add updates but each step is launch-bound (dt won't shrink below ~8ms), so it would cut images/epoch with no dt benefit — also unpromising. Batch size is bracketed: 128 is at/near the throughput-optimal point for this launch-bound net.

## Next Steps
- **Bag-of-Tricks free convergence levers (zero-init residual γ + no-bias-decay)** — the brainstorm-025 candidate #2: compute-NEUTRAL, param-neutral, keeps the 91-epoch baseline (no epoch-wall / update-collapse risk), directly targets the convergence-bound regime. (medium confidence it's a small gain; low-risk.)
- **Per-channel input std-normalization** (std=(1,1,1)→true CIFAR std) — the last cheap compute-neutral scalar probe; expected BN-absorbed null, axis-closer. (medium confidence clean null; low confidence gain.)
- After those two compute-neutral probes, the search is genuinely exhausted: ~18 axes closed, all scalar knobs bracketed (now including batch size), every compute-adding change hits the epoch wall, every batch increase is compute-bound and collapses updates, every added regularizer under-fits. The honest scientific conclusion is the **96.22 plateau is the ceiling for k=4 ResNet-20 at 300s/H20**. (high confidence the plateau is real.)

## Exit Action Results
<!-- No exit actions defined for this goal. -->
