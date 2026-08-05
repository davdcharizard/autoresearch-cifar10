# Report EXP-009: Muon optimizer (Newton-Schulz orthogonalized momentum) on the conv weights
- **Created**: 2026-06-29

## Goal
Maximize CIFAR-10 `best_test_acc` (%) within the fixed 300s training budget, editing only `train.py`. Baseline (EXP-008): **96.38%**; improvement bar **≥96.48%** (+0.10pp) AND clearly above the ~0.1pp noise floor. Higher is better.

## Idea & Hypothesis
Chosen idea (EXP-009 brainstorm, cross-model-review pick): replace SGD-Nesterov on the conv weights with **Muon** — SGD momentum whose update matrix is orthogonalized by a fixed Newton-Schulz quintic before application. Muon is the optimizer behind Keller Jordan's newest fast-CIFAR records (airbench94_muon) on this exact whitened wide-shallow ResNet family. Hypothesis: orthogonalized, better-conditioned conv-weight updates would (a) complete the EXP-008 under-annealed tail and (b) reach a better-generalizing minimum, clearing 96.48 at near-zero throughput cost. The cross-model idea-review picked Muon over the two safer riders (decoupled-WD, EMA-retune) precisely because it was the only finalist with a credible above-noise ceiling under one-shot fixed-seed evaluation.

## Approach
Implemented a from-scratch torch-only Muon in `train.py` (no new dep), grounded in the canonical `airbench94_muon.py` (fetched during planning) and the idea-review's weight-decay-correctness concern:
- `zeropower_via_newtonschulz5` (bf16 quintic, coeffs (3.4445,−4.7750,2.0315), ns_steps=3, transpose-if-tall) + `class Muon`.
- **Conv weights only** (`p.ndim==4`, 10 tensors) → Muon at peak LR **0.24** (airbench's proven value), momentum 0.9 (shared, unchanged), **airbench weight re-normalization** `p←p·√out/‖p‖` replacing L2 decay (resolves the review's "coupled-WD-through-orthogonalization" concern by removing WD from the Muon group entirely; safe because every conv is BN-followed → scale-invariant).
- **fc/BN/α** (22 tensors) → the **unchanged EXP-008 SGD-Nesterov** (lr 0.4, mom 0.9, wd 5e-4).
- Both groups driven off the shared time-based triangular one-cycle envelope; EMA/whitening/TTA/augmentation/seeds byte-identical to EXP-008.
- Plan hardened by a 12-point cross-model plan-review (fixed a guaranteed step-50 `NameError` from the removed `lr` var, added budget-used/anti-bookkeeping/true-scope verification gates, and an img/s-based throughput gate to separate Muon overhead from host contention). Milestone-1 CUDA smoke test passed (NS singular values 0.83–1.20, renorm fired).

## Execution
Single run, no retries. Launched `CUDA_VISIBLE_DEVICES=1 uv run train.py` under `timeout 600` on GPU 1; completed cleanly (exit 0) in 440.4s wall, 300.0s training, 138 epochs / 13378 steps, peak VRAM 1635 MB. Param split confirmed correct (Muon 10 conv tensors + SGD 22 tensors), num_params 7,784,627 unchanged. Throughput steady ~23.4–23.7k img/s (≈EXP-008 band; ns_steps=3 overhead negligible). No NaN/inf, no crash.

## Results
- **Primary metric**: **94.11%** (baseline 96.38, delta **−2.27pp**, −2.36%). Bar 96.48 not met.
- **Observations**: The trajectory is a **divergence-and-recovery**, not a healthy curve. Test acc climbs to 77.74% by ep3, then **collapses to ~random** for the entire high-LR phase — ep10 53.6%, **ep25 10.00%** (pure random), ep50 12.4%, ep75 15.7%, ep100 19.5% — and only **re-learns as the LR anneals toward 0**: ep120 76.6% → ep138 94.11% (best==final, still rising). 25 epochs showed >0.5pp test-acc drops. The collapse to exactly 10% = the network output degenerated (e.g. near-constant predictions) during the high-LR plateau.
- **Analysis**: This is the **"Muon peak LR too high"** signature. With airbench's weight-renorm pinning ‖conv‖=√out, the per-step update at peak (lr 0.24 · ‖orthogonalized update‖ ≈ 0.24·√512 ≈ 5.4 vs ‖p‖≈22.6) is a **~24% rotation of every conv weight matrix per step** — sustained over the long high-LR plateau of our 150-epoch one-cycle, this destabilizes the BN statistics and collapses the net. airbench tolerates 0.24 because its schedule is only **8 epochs** with **no high-LR plateau** (linear decay from peak, no warmup) — the net never sits at high LR long enough to diverge. Our triangular one-cycle holds near-peak LR for many more epochs, so the *same* peak that is safe in an 8-epoch sprint is far too hot here. The clean late recovery to 94.11 (and still rising) proves the **architecture, data pipeline, weight-renorm, NS numerics, EMA, and dual-optimizer wiring are all correct** — the failure is purely the LR magnitude / schedule-length interaction, exactly one knob.
- **Key Learning**: Muon's airbench peak LR 0.24 does NOT transfer to our long (~150-epoch) one-cycle — the prolonged high-LR plateau makes the orthogonalized+renormed conv update (~24%/step rotation) diverge; the optimizer/implementation is sound, the peak LR must drop ~2–3× (to ~0.08–0.12).

## Verification
- **Conditions**: NC1 PASS (completed in budget, valid metric, training_seconds=300.0, wall 440s<600s, exit≠124). **NC2 FAIL** (94.11 < 96.48; anti-bookkeeping check passed — summary best == max per-epoch trace 94.11, no tampering). NC3 recorded clean (only `M train.py`, prepare.py byte-unchanged, num_params unchanged, seeds intact, ≤1 eval/epoch) — no scope/integrity issue.
- **Review Notes**: Results trustworthy. Valid metric genuinely produced; the loss stayed finite throughout (collapse was to random ~10%, loss ~2.29, not NaN), so this is a real training outcome, not an infra failure. The single change vs EXP-008 (conv optimizer package) is cleanly attributable.
- **Verdict**: **no-improvement**
- **Verdict Basis**: valid result, primary metric did not exceed baseline (NC2 failed); no hard-constraint violation (not invalid), results produced (not crash).

## Unexplored Avenues
- **Lower Muon peak LR (the direct fix)**: PEAK_LR_MUON 0.24 → ~**0.08–0.12**, all else identical. The divergence-and-recovery shows the net *can* learn under Muon (it reached 94.11 once LR fell); a lower peak should keep it stable through the high-LR phase and let it anneal from a much higher floor. Highest-information single follow-up.
- **Longer/​gentler warmup or a lower-plateau schedule for the Muon group**: increase PCT_START for the Muon group, or cap the Muon LR with a flatter envelope, so the orthogonalized update never sits at the destabilizing ~24%/step rotation. Addresses the schedule-length root cause rather than just the peak.
- **Fewer/decoupled effective step size via the update-scale convention (proposal idea-03's original variant)**: instead of weight-renorm (which fixes ‖p‖ and makes the rotation large), use the `sqrt(out/in)` update-scale convention with decoupled WD and a small LR (~0.02) — the proposal's original design, which the review steered away from for LR-grounding reasons but which may be *more* stable here because ‖p‖ can shrink under WD, reducing per-step rotation.
- **ns_steps / momentum retune**: not the bottleneck here (smoke + recovery show NS is fine), but once a stable LR is found, momentum (airbench 0.6 vs our 0.9) is a secondary knob.

## Next Steps
1. **EXP-010 — Muon with PEAK_LR_MUON ≈ 0.10** (high confidence this is the right lever): re-run the *identical* recipe with only the Muon peak LR lowered ~2.4× to kill the high-LR divergence; read the same trajectory diagnostics (ep10/ep25 must now stay ≳85/≳92 instead of collapsing). This is the pre-registered LR-too-high follow-up and the single highest-value next move.
2. **If still unstable at 0.10** (medium): switch the Muon group to the update-scale convention + decoupled WD at lr ~0.02 (proposal's original variant), which lets ‖p‖ shrink and reduces per-step rotation — a structurally gentler Muon.
3. **Fallback if Muon proves not worth the tuning budget** (medium): pivot to the two deferred throughput-free riders that the idea-review ranked as sound-but-small (decoupled-WD on BN/α, EMA-horizon 0.998→0.995) — lower ceiling but near-zero risk, and they compose with the proven EXP-008 recipe.
