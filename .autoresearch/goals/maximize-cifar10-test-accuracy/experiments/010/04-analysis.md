# Report EXP-010: Muon conv-weight optimizer — PEAK_LR_MUON sweep
- **Created**: 2026-06-29

## Goal
Maximize CIFAR-10 `best_test_acc` (%) within the fixed 300s training budget, editing only `train.py`. Baseline (EXP-008): **96.38%**; bar **≥96.48%** (+0.10pp) AND clearly above the ~0.1pp noise floor. Higher is better.

## Idea & Hypothesis
Parameter sweep (EXP-009 follow-up). EXP-009 showed the from-scratch Muon optimizer (Newton-Schulz orthogonalized momentum on the conv weights, airbench weight-renorm, ns=3) is implemented correctly but **diverged at peak LR 0.24** (collapse to ~random, recovering only in the anneal tail → 94.11). The trajectory was an unambiguous "LR too high" signature. Hypothesis: the Muon method is sound and a **lower peak LR** will train stably and, per Muon's selling point, reach a *better minimum* than tuned SGD-Nesterov — clearing 96.48. The sweep tunes the single open variable, `PEAK_LR_MUON`.

## Approach
Swept `PEAK_LR_MUON ∈ {0.03, 0.06, 0.10, 0.14}` (grid, anchored below the divergent 0.24), all else byte-identical to the EXP-009 Muon recipe (conv→Muon ns=3 weight-renorm no-WD; fc/BN/α→unchanged EXP-008 SGD lr 0.4 wd 5e-4; EMA/whitening/TTA/aug/seed fixed). `train.py` made `PEAK_LR_MUON` env-overridable so trials set it at runtime without editing the file; `sweep.py` ran each as a full-budget trial (real best_test_acc) and emitted `trials.tsv`/`best.json`. Winner applied to `train.py` (default 0.03) and confirmed at full throughput. Full details: `02-sweep.md`, `01-brainstorm.md`.

## Execution
- **4 grid trials** ran while **GPU 1 was externally contended** (~18.5 GB foreign job, 100% util) → throughput halved (~9.8k img/s, 59–64 epochs vs normal ~150). Absolute trial accuracies are therefore throughput-confounded (under-annealed); only the *ranking* (all trials equally slowed) is valid. Ranking was clean and monotonic: **0.03 > 0.06 > 0.10 > 0.14** (95.90 > 95.54 > 94.42 > 93.33 at ~60 ep) → winner **0.03**, the lowest grid point; all stable (none diverged, unlike 0.24).
- **Winner confirmation** at restored full throughput (user confirmed GPU 1 free): `PEAK_LR_MUON=0.03`, full 300s, 132 epochs, ~24.5k img/s. Clean stable trajectory, no divergence.

## Results
- **Primary metric**: **96.33%** (baseline 96.38, delta **−0.05pp**, −0.05%). Below the 96.48 bar.
- **Observations**: The lower LR completely fixed EXP-009's divergence — the confirmation trained cleanly and monotonically: ep25 89.66 → ep50 92.15 → ep75 94.65 → ep100 95.81 → ep125 96.20 (best 96.33) → ep132 96.19. It **peaked at ep~125 then slightly dipped** (i.e. fully annealed/converged — the schedule is time-based so LR→0 at the budget end regardless of the 132 vs 150 epoch count; not truncated/under-annealed). Versus EXP-008 SGD at matched epochs: Muon-0.03 is **slower early** (ep25 89.66 vs SGD ~92.3; ep50 92.15 vs ~93.75) but **catches up and edges ahead mid-run** (ep100 95.81 vs SGD ~95.13), finishing at essentially the same level (96.33 vs 96.38).
- **Analysis**: The hypothesis's *stability* half is confirmed (0.03 trains cleanly); the *better-minimum* half is **falsified** — optimally-tuned Muon only **matches** tuned SGD-Nesterov on this net/budget (96.33 vs 96.38, a tie within the ~0.1pp noise floor), it does not beat it. The monotonic lower-is-better ranking with the winner at the grid floor (0.03) hints the optimum could be marginally lower, but 0.03 is already slower early (an even lower LR would under-anneal more of the early schedule, not help), so the Muon ceiling here is ~SGD's. This matches the plan's pre-registered falsification: "trajectory stable AND fully annealed yet best_test_acc < 96.38 → Muon helped convergence but not generalization on this net; lever exhausted."
- **Key Learning**: A correctly-tuned orthogonalized optimizer (Muon, best LR 0.03) only MATCHES tuned SGD-Nesterov on this whitened ResNet-9 at the 300s budget (96.33 vs 96.38) — the optimizer-swap lever is effectively exhausted; the recipe is regularization-bound, not optimizer-bound.

## Verification
- **Conditions**: NC1 PASS (full-throughput confirmation: completed, training_seconds 300.0, exit 0, valid metric 96.33%, wall 445.7s). **NC2 FAIL** (96.33 < 96.48; −0.05 vs baseline, within noise; anti-bookkeeping passed — max trace 96.33 == summary 96.33). NC3 PASS (only `M train.py`, prepare.py byte-unchanged, num_params 7,784,627 unchanged, seeds intact, ≤1 eval/epoch).
- **Review Notes**: Trustworthy. The trial-phase throughput confound is handled correctly (ranking used, winner confirmed at full throughput); the confirmation is a fair, fully-annealed comparison. Genuine no-improvement, not invalid (no scope/integrity issue) and not crash (valid metric).
- **Verdict**: **no-improvement**
- **Verdict Basis**: valid result, primary metric did not exceed baseline by the required margin (NC2 failed); no hard-constraint violation, results produced.

## Unexplored Avenues
- **Even lower PEAK_LR_MUON (≤0.02) or a Muon-specific longer warmup**: the ranking floored at 0.03; a touch lower or a gentler ramp *might* lift the early-epoch deficit. Low confidence it crosses the bar — 0.03 already converges to SGD's level and lower LR worsens the early under-convergence.
- **Muon + decoupled-WD (update-scale convention, no weight-renorm)**: EXP-009/010 used airbench weight-renorm; the proposal's original variant (update scaled by √(out/in), ‖p‖ free, decoupled WD ~lr-matched) is a structurally different Muon that *might* find a different (better?) minimum. Medium-low confidence given Muon already ties SGD.
- **Muon momentum retune (0.9→0.95) or ns_steps**: secondary knobs; unlikely to convert a tie into a clear win.
- **The optimizer axis appears largely exhausted** — both SGD (proven) and Muon (this) land at ~96.35±noise. The productive lever class remains throughput-free *regularization* (EXP-008's +0.38pp win), not the optimizer.

## Next Steps
1. **Abandon the optimizer axis; return to throughput-free regularization / eval-side levers** (high confidence this is where headroom is): the two idea-review-sound riders deferred from EXP-009 — EMA-horizon retune (0.998→0.995) and decoupled-WD on BN/α — plus further augmentation tuning, all compose on the proven EXP-008 SGD recipe (96.38) and target the *regularization-bound* limiter rather than the now-tied optimizer.
2. **(Low priority) One more Muon shot at PEAK_LR_MUON≈0.02 with longer warmup** only if regularization levers stall — low confidence it beats the tie.
3. **Consider revisiting capacity at the proven 8×8 stage with a milder step** (256→320, EXP-007 follow-up) now that throughput is the recurring confound — medium confidence, but watch num_epochs.
