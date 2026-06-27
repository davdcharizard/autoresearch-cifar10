# Report EXP-062: Schedule-Free SGD — full anneal→averaging substitution, eval at x
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-062.md
- **Plan**: plans/plan-062.md
- **Log**: logs/exp-log-062.md

## Goal

Maximize CIFAR-10 best_test_acc (%, higher is better) within the fixed 300s charged budget by modifying train.py only. Baseline: 96.71 @ 1990397; bar ≥ 96.81 (family mean 96.57, σ 0.16). Specific question: after 55 closures emptied every catalogued axis, brainstorm-062's literature excavation surfaced the one optimizer-schedule construction absent from the record — does Schedule-Free optimization (Defazio et al., NeurIPS 2024), whose CIFAR-10 SGD evidence claims at-or-above the ENTIRE cosine-horizon frontier, beat the time-keyed cosine at the 300s horizon?

## Idea & Hypothesis

Chosen from brainstorm-062: replace the cosine anneal + nesterov SGD entirely with **Schedule-Free SGD** — train at constant lr 0.4 after a 2000-step warmup; maintain z (SGD iterate) and x (lr²-weighted online average); evaluate gradients at y = 0.1z + 0.9x; eval each epoch at x with BN running stats refreshed at x (24 forward-only batches from a second loader, uncharged). Distinct from the EXP-011/032 closures: those changed the READOUT of an annealed trajectory; this changes the GENERATING process (z never anneals). Hypothesis: x holds at-or-above the cosine endpoint while its smooth monotone curve extends the near-ceiling plateau ~6× → ≥ 96.81; pre-registered alternates: (ii) family band → schedule axis closes at the family level; (iii) < 96.41 → constant-lr basin/convergence loss the averaging cannot repair; (v) BN-mismatch signature → engineering retry.

## Approach

Five hunks to train.py: constants (WARMUP_STEPS 2000, SF_MOMENTUM 0.9, WEIGHT_LR_POWER 2.0, SF_R 0, REFRESH_BATCHES 24; PEAK_LR 0.4 retained as the constant lr); `lr_at` deleted and a reference-faithful `ScheduleFreeSGD` class added (params hold y; foreach step: WD-at-y → y.lerp_(z, ckp1) → y += grad·lr(β(1−ckp1)−1) → z −= lr·grad; guarded eval_mode/train_mode in-place y↔x lerp switches); same two selective-WD param groups; second persistent refresh_loader; eval block does eval_mode → 24 bf16 forward-only refresh batches → evaluator.evaluate (unchanged) → train_mode, all outside the timed window. CPU sanity 14/14 (algebra exact to 4e-16 vs an fp64 reference of the published equations; roundtrip 6e-8; selective WD verified). Exact math distilled in knowledge/papers/schedule-free-road-less-scheduled.md.

## Execution

One GPU probe, one clean run, zero retries:
- Probe (load 33): P = 23.28ms → LAUNCH branch; refresh R = 0.306s/24 batches → REFRESH_BATCHES kept at 24.
- Run 1: gates poll 1 (load 13); D0 22.5; windows 21.4–23.0, slow_streak 0; RC=0. Ledger: **13,471 steps / 139 epochs — exactly the FAMILY signature (13,100–13,600 @ 22.0–22.8ms)**: the schedule-free foreach step is dt-FREE, and the probe's 23.28 was load-inflated (probe-to-run offset INVERTED, −1.0ms, at probe load 33). VRAM 1,613.0MB (exactly family). Total 601.5s — 1.5s over the 600s cap, wall-side only (139 × ~1.35s eval+refresh; charged 300.0s exact; EXP-013 precedent applies). ep1 22.05 (below the relaxed tripwire — the genuine x-at-init signature; ledger clean, no contamination).

## Results

- **Primary metric**: best_test_acc 94.87 @ ep137 (baseline: 96.71, delta: −1.84, −1.90%)
- **Observations**: The x-curve delivered the paper's SHAPE perfectly — test_loss strictly monotone 2.13 → 0.229, accuracy monotone, zero plateau scatter — but the LEVEL is catastrophically behind: still climbing at budget end (new best at ep137/139, loss still falling over the final 10 epochs), reaching 94.87 where the cosine reads ~96.6 at the same step count. Only 3 evals within 0.15 of best (family 8–12; the plateau-length mechanism predicted ≥30) — there is no plateau, only the tail of a slow climb. test_loss 0.229 vs family 0.185. BN-mismatch ruled out (no depressed-gap signature; refresh verified by the smooth curve itself).
- **Analysis**: Pre-registered branch (iii) fired at 12σ depth — not a noise question. The result refutes the paper's any-horizon claim FOR THIS REGIME: at a 300s/139-epoch horizon under heavy augmentation, explicit lr²-weighted averaging of a hot constant-lr trajectory converges far slower than the annealed trajectory it replaces. The mechanism reading: the cosine's decaying lr is doing real optimization work the averaging cannot reproduce — late-phase small steps REFINE the basin (curvature-adaptive step sizes), whereas averaging hot iterates merely cancels their noise around a basin the hot z-path never refines. This is EXP-016's lesson (the tail's job is convergence manufacturing) in its strongest form: removing the anneal entirely costs −1.84, dwarfing the −0.50 of merely reshaping it to linear. EXP-032's "the anneal already performs implicit averaging" now has its converse measured: averaging does NOT perform implicit annealing. The schedule axis is closed at the FAMILY level — both within the anneal family (EXP-010/014/016/049) and now across families (anneal vs averaging-based schedule-free). Fifty-sixth consecutive closure; external transfer 0-for-18, and this failure is a new mechanism class for the transfer record: horizon-convergence mismatch (the paper's CIFAR figures are step-budget regimes whose horizons all sit past the averaging curve's catch-up point; our 13.5k-step horizon sits well before it).
- **Key Learning**: The anneal is load-bearing in its own right — schedule-free's theorem-backed schedule/averaging equivalence does not hold at short horizons where the averaged hot trajectory has not yet caught the annealed one; "matches every cosine horizon" claims must be checked against the horizon's position on the catch-up curve.

## Verification

- **Conditions**: Integrity pre-condition PASSED (pristine telemetry; FAMILY ledger binding — probe-revised bands rejected as load-inflated, fast-side deviation integrity-benign; params exact; ep1 explained by x-at-init with clean ledger; BN-mismatch screen negative). Condition 1 FAILED: 94.87 < 96.81 (branch iii). Condition 2 FAILED informationally: total 601.5s > 600 (wall-side +1.5s from refresh+eval; charged 300.0s exact; EXP-013 precedent — metric recorded, overage noted; a 12-batch refresh would shave ~30s but cannot move a −1.84 read). Condition 3 pass (139/139).
- **Review Notes**: Results trustworthy — watchdog full coverage, family-ledger agreement, the deficit came through the intended intervention class, and the trajectory shape independently confirms the mechanism engaged (smooth monotone x-curve ≠ any prior failure signature).
- **Verdict**: no-improvement
- **Verdict Basis**: Condition failure — valid below-bar result at pre-registered branch (iii).

## Unexplored Avenues

- **Larger constant lr (paper guidance: schedule-free optimum often ≥ scheduled peak)**: the one genuinely open variant — a hotter z explores faster and x might catch up sooner. But the deficit is −1.84 and the record's largest retune recovery is ~0.5; the heat closures (EXP-010: +50% peak → −0.57 annealed) say hotter trajectories lose basin quality here. Expected value low; a single probe-style short run could bound it cheaply if ever revisited.
- **Schedule-free AdamW**: the paper's stronger results are AdamW on large-scale tasks; but Adam-family geometry is unevidenced on this small-CNN/SGD-tuned recipe and the convergence-horizon objection applies identically. Low.
- **Hybrid (schedule-free until p≈0.85, then anneal the y/z pair)**: reintroduces the anneal — interpolates back toward the measured optimum, inherits EXP-032's carve-out objection. Closed by composition logic.

## Next Steps

The deepest remaining honest move (literature excavation) was executed and its one surviving candidate refuted at 12σ. Status for brainstorm-063: (a) 56 consecutive closures; the external frontier is empty again — the excavation protocol itself is now validated as able to surface genuinely new candidates (schedule-free was real, untested, screen-passing), so further sweeps remain legitimate but the prior drops with each emptied vein (confidence low); (b) the measured-ceiling hypothesis stands unfalsified and strengthened — even a NeurIPS-2024 theorem-backed schedule alternative lands 1.84 below; (c) remaining honest moves are unchanged: deeper/adjacent excavation (e.g., 2025–2026 venues via /lit-search rather than ad-hoc web search; low), genuinely novel non-interpolation compositions passing the +0.3pp screen (none currently constructible; low), or plateau-length micro-harvests (fail the screen; reject). Brainstorm-063 must state which it attempts and why its candidate is not a re-measurement.

## Exit Action Results
