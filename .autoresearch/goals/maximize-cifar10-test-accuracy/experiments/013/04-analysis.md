# Report EXP-013: Tail-only Sharpness-Aware Minimization (SAM)
- **Created**: 2026-06-29

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within the fixed 300s training budget, editing
only `train.py`. Baseline **96.38%** (EXP-008, commit 07c3760); improvement bar **≥96.48** (+0.10pp,
clearly above the ~0.1pp throughput-jitter noise floor).

## Idea & Hypothesis
Chosen (thorough brainstorm, Codex-reviewed pick over mild-capacity and cosine-schedule): **tail-only
Sharpness-Aware Minimization**. The net is regularization-bound near its generalization ceiling and
every regularizer that reallocates the existing knobs (input-aug, weight-decay, label smoothing,
optimizer) has saturated within noise (EXP-009–012). SAM attacks generalization through a different
mechanism — it steers SGD toward FLAT minima (lower worst-case neighborhood loss) by descending on the
gradient computed at an ascent-perturbed point `w + ρ·g/‖g‖` — and the EXP-011 learnings entry
explicitly names loss-geometry/SAM as the next-different-mechanism. Published SAM gains on CIFAR ResNets
are +0.3–1.0pp at matched epochs (3–10× the noise floor). The dominant cost (2× fwd-bwd → under-anneal)
was controlled by applying SAM only in the low-LR tail where this recipe's accuracy concentrates
(EXP-001), keeping the global epoch count near ~124 (> the ~110 under-anneal cliff).

**Hypothesis**: SAM (ρ=0.05) in the final 35% of the budget (`progress ≥ 0.65`) raises best_test_acc
≥0.10pp over a same-session baseline and clears 96.48, at num_epochs ≥115. Falsifiable: if epochs hold
but accuracy ties/loses, the flat-minima gain doesn't survive the reduced tail-step count on this small
net at 300s.

## Approach
All changes in `train.py` only (no new deps). Env-toggled `SAM_RHO` (0.0=off→baseline) and
`SAM_START_FRAC`. A module-level `sam_step()` does the two-pass SAM: 1st fwd-bwd at `w` → ascent
`e_w=ρ·g/‖g‖` on **fp32 master params** (autocast casts only the forward; bf16-instability mitigation) →
2nd fwd-bwd at the perturbed point with BN running buffers frozen via `track_running_stats=False` in a
`try/finally` (so the perturbed pass touches NO BN buffer — running_mean/var AND num_batches_tracked —
which the weight-EMA averages) → restore `w` → `optimizer.step()` applies Nesterov/momentum/wd to the
perturbed gradient. The loop gates `use_sam = SAM_RHO>0 and progress ≥ SAM_START_FRAC`. A pre-run smoke
exercised the real `sam_step` and asserted optimizer-stepped, BN buffers +1 per step (perturbed pass
inert), flag restoration, restore-within-tolerance, and gate logic. Three same-session cells (EXP-012
pattern). Plan- and idea-reviews (Codex) caught and fixed the critical gate bug (`progress≥0.65` not
`≥0.35`), the BN-momentum-vs-track_running_stats subtlety, and mandated win-confirmation; all applied.

## Execution
All three cells ran back-to-back in one background process (exit 0), GPU 1 uncontended, ~26.6k img/s in
the plain-SGD phase. **No NaN/inf anywhere** — the fp32-perturbation ascent was numerically stable under
bf16 autocast (the load-bearing correctness assumption held). No retries. The gate fired exactly as
designed: cell-A SAM-active for 2670 steps (22.3% of the step count = the final-35%-of-TIME tail at 2×
cost), cell-B 1908 steps (15.0%). Epochs landed at the predicted ~124/~132.

## Results
- **Primary metric**: 96.29% (best SAM cell, cell-A) (baseline: 96.38, delta: −0.09; vs **same-session**
  baseline 96.47, delta −0.18).
- **Table** (best / final / epochs / sam_step_frac):
  - cell-0 (SAM off): **96.47** / 96.47 / 150 / 0.000  — a strong same-session draw (+0.09 over stored 96.38, within the noise floor)
  - cell-A (ρ0.05, final 35%): **96.29** / 96.29 / 124 / 0.223  → **−0.18pp vs cell-0**; best==final, monotone-rising to ep124
  - cell-B (ρ0.05, final 25%): **96.18** / 96.11 / 132 / 0.150  → **−0.29pp vs cell-0**; peaked ep129 then dipped (annealed)
- **Observations**: SAM achieved its intended LOCAL effect (it ran, was stable, fired in the tail, cost
  the predicted epochs) but produced NO accuracy gain — both SAM cells fell clearly below the
  same-session baseline. cell-A's tail rose monotonically to the last epoch (96.10→96.29, best==final) —
  the classic under-anneal signature: the 26 epochs SAM's 2× cost removed left the net still climbing.
  cell-B, the lighter variant, ANNEALED cleanly (peaked then dipped) at 132 epochs and still lost by
  0.29pp — so under-anneal is fully controlled for, and SAM provides zero lift even when the schedule
  completes. The two cells bracket the SAM-active fraction (22% vs 15% of steps); more SAM (cell-A) is
  marginally higher than less (cell-B), but both sit ~0.2–0.3pp below baseline — there is no SAM fraction
  that lifts above it, and lighter-still would only approach baseline from below.
- **Analysis**: hypothesis rejected. The flat-minima generalization gain — real on heavier backbones at
  matched epochs — does NOT survive the epoch cost on this small wide-shallow net under a 300s time
  budget. The mechanism is sound and correctly implemented (stable, no NaN, exact BN handling), but the
  time-budgeted setting converts SAM's 2× per-step cost directly into ~26 fewer anneal epochs, and on a
  net where accuracy concentrates in the low-LR tail that epoch loss dominates any basin-selection
  benefit. This is the SAME under-anneal tradeoff that sank the capacity experiments (EXP-005/007),
  arriving now via compute-per-step instead of params-per-step.
- **Key Learning**: tail-only SAM ties-to-loses (best 96.29 vs same-session 96.47) — its 2× step cost
  removes ~26 anneal epochs (150→124) and the flat-minima gain doesn't offset that at 300s; even the
  cleanly-annealed lighter cell (96.18) is below baseline, so the loss is the cost, not under-anneal.

## Verification
- **Conditions**: NC1 PASS (all cells 300s training, <600s wall, valid metrics, exit 0, no NaN). **NC2
  FAIL** (best SAM cell 96.29 < 96.48 and −0.18pp below same-session cell-0; no win → no confirmation
  re-run). NC3 PASS (only `M train.py`, prepare.py unchanged, num_params 7,784,627, seeds intact, 1 eval/epoch).
- **Review Notes**: results trustworthy — same-session control reproduced the recipe at a normal draw,
  anti-bookkeeping passed (max per-epoch == summary best, all cells), SAM instrumentation
  (sam_steps/frac) internally consistent with the time-based gate, no instability. The under-anneal
  diagnostic was explicitly controlled by the lighter annealed cell-B.
- **Verdict**: no-improvement
- **Verdict Basis**: valid, stable result; NC2 (result quality gate) failed — both SAM cells below the
  same-session baseline.

## Unexplored Avenues
- **SAM in the plain-SGD step WITHOUT the 2× cost** — e.g., ESAM-style sparse weight perturbation (perturb
  a random ~50% subset each step) or LookSAM-style reuse of the ascent direction across k steps. These cut
  the cost so SAM can run over MORE of training without halving epochs. Honest caveat: the EXP-013 result
  shows zero positive SAM signal even where it ran, and the literature finds periodic/sparse SAM *loses*
  accuracy — so the expected value is low; only worth it if a near-free SAM variant exists.
- **A genuinely faster base step to "buy back" the SAM epochs** — SAM only makes sense here if per-step
  throughput rises enough that full SAM still fits ~150 epochs. No such headroom is known (the net already
  runs at ~26.6k img/s, compute-bound). Not promising within the constraints.
- **The capacity and schedule levers (EXP-013 finalists not chosen)** remain genuinely untried: mild
  layer2 256→320 (idea-01) and cosine/earlier-peak schedule (idea-02). Both avoid SAM's 2× cost.

## Next Steps
1. **One-cycle cosine + earlier-peak schedule reshape** (confidence: low-medium) — the cheapest untried
   throughput-free lever (idea-02, already fully proposed); cannot under-anneal. Honest upside ~+0.05–0.15pp
   (coin-flip on the bar), but zero downside and the last untouched training-side axis. Good next probe.
2. **Mild capacity widen layer2 256→320** (confidence: medium) — highest remaining ceiling (idea-01,
   proposed); a different axis (capacity). Under-anneal risk (~122–128 ep predicted), so pre-register
   num_epochs; if it under-anneals like SAM/EXP-007, the capacity axis is then strongly exhausted.
3. **Accept proximity to ceiling** (confidence: medium) — 7 straight no-improvements (EXP-006–013) across
   optimizer, eval-TTA, input-aug, weight/target regularization, and now loss-geometry all land within
   ~0.1pp of ~96.4. After the two cheap levers above, the evidence for a real generalization ceiling at
   ~96.4–96.5 for this net/budget is strong; further gains likely need a different base architecture.

## Exit Action Results
- None defined (autopilot goal) — section intentionally empty.
