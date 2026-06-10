# Report EXP-020: SWA with a lower constant-LR floor (SWA_LR 0.05 → 0.02)
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-020.md
- **Plan**: plans/plan-020.md
- **Log**: logs/exp-log-020.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within the fixed 300s budget on
a single H20. Baseline = **96.22%** (EXP-012, commit 6c417a4); success bar = **96.32%** (+0.1pp). This experiment
refined the EXP-019 SWA near-miss by lowering the constant-LR averaging-tail floor.

## Idea & Hypothesis
Chosen idea: re-run EXP-019's proper-SWA setup EXACTLY (cosine PEAK_LR→floor over [5%,75%], constant floor for the
final 25%, per-epoch weight averaging, truncated BN-recompute, SWA model evaluated in the tail), changing ONLY the
floor LR 0.05 → 0.02. Selected because EXP-019 was a −0.25pp near-miss whose diagnosis pinned the floor as too high
(raw iterate at 0.05 cratered to 91.8%, capping snapshot top-1 quality). Hypothesis: a lower floor yields
higher-top-1 snapshots while retaining enough iterate movement for flat-region averaging → best_test_acc above
EXP-019's 95.97 and plausibly past 96.32; if instead it lands ~96.0–96.1 (too little movement → approaching a
single constant-0.02 endpoint without cosine-to-0 sharpening), the SWA floor sweet spot doesn't clear cosine-to-0.

## Approach
One train.py change vs the validated EXP-019 code: `SWA_LR = 0.05 → 0.02` (SWA_START_FRAC=0.75,
BN_RECOMPUTE_BATCHES=50, and the full EXP-012 recipe all unchanged). The SWA machinery (AveragedModel,
`recompute_bn`, tail-eval branch, modified `lr_at_fraction`) is byte-identical to EXP-019 otherwise. No new deps
(`swa_utils` is core torch). No deviations from plan-020. Ruff clean; diff = train.py only.

## Execution
One run, no retries. Clean startup: `num_params 4,299,866` (UNCHANGED), clean compile, no traceback, no NaN. LR
verified: decayed 0.20→0.02 then held EXACTLY `lr: 0.0200` constant through the tail (ep 68–91). Tail fired at
ep 68 (~75%): 67 `[raw]` + 24 `[swa]` evals = 91 = num_epochs (one evaluate()/epoch, constraint satisfied).
Throughput-neutral: 91 epochs (= EXP-012/EXP-019). Exited 0 in 422.0s < 600s. peak VRAM 469.3 MB.

## Results
- **Primary metric**: best_test_acc = **96.13%** (baseline 96.22, delta **−0.09pp**, −0.09%) — below the 96.32
  bar, but **+0.16pp over EXP-019's 95.97**. final_test_loss 0.1806 (≈ EXP-019's 0.1788, far below baseline 0.195).
- **Observations**: Lowering the floor 0.05→0.02 lifted SWA accuracy as predicted, CONFIRMING the EXP-019
  diagnosis (the 0.05 floor was too high). The SWA-eval curve was again STILL RISING at the budget end
  (ep 89→90→91: 96.06→96.08→96.13) — the average had not fully converged.
- **Analysis**: Hypothesis partially confirmed (floor↓ → acc↑) but the bar was not cleared, and the result is now
  DECISIVE for the axis. The SWA floor sweep reads **0.05 → 95.97 (EXP-019), 0.02 → 96.13 (EXP-020)** — monotone
  improvement as the floor drops, approaching the cosine-to-0 baseline (96.22) FROM BELOW. This is structural, not
  coincidental: as SWA_LR → 0 the constant-LR tail degenerates into "cosine-anneal-to-~0 then hold ~0", i.e. the
  baseline schedule itself, so the SWA result's supremum over the floor is exactly the 96.22 cosine-to-0 endpoint.
  SWA therefore cannot EXCEED cosine-to-0 on this top-1 metric at this budget — it only recovers it in the limit.
  Both SWA runs produced the project's lowest test losses (~0.18 << 0.195), i.e. genuinely flatter/better-calibrated
  minima, but flatness/loss did not translate to higher top-1 (a known SWA signature). This closes the
  weight-averaging axis (EXP-006 EMA no-op; EXP-019/020 SWA approaches-but-doesn't-beat cosine-to-0) and
  re-corroborates the standing diagnosis: the 96.22 plateau is generalization-bound at fixed k=4 capacity in 300s,
  and a well-tuned cosine-to-0 schedule is already near the top-1 ceiling for this recipe.
- **Key Learning**: Lowering the SWA floor 0.05→0.02 lifted the average 95.97→96.13 (confirming the floor-too-high
  diagnosis), but SWA monotonically approaches the cosine-to-0 baseline (96.22) from below and cannot exceed it —
  weight-averaging axis is CLOSED for this top-1 metric/budget.

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.32) **FAIL** (96.13, −0.09pp) — decisive. Cond 2 (clean completion
  < 600s, no traceback) PASS (total_seconds 422.0). Cond 3 (scope) PASS (train.py only, eval-count 91 == num_epochs,
  params 4,299,866 unchanged, no new deps, seed 42 intact).
- **Review Notes**: Results trustworthy — clean run, throughput-neutral fair test (params unchanged, 91 epochs ==
  EXP-012/EXP-019), SWA path verified (constant 0.0200 LR, monotone SWA-eval climb), scope intact. Single-variable
  change from EXP-019 gives clean attribution of the +0.16pp to the lower floor.
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (primary metric did not clear the bar; −0.09pp below baseline).

## Unexplored Avenues
- **Even-lower floor (0.01) / different start-frac**: would inch closer to 96.22 but, by the asymptote argument
  above, cannot exceed it — negative expected value, NOT worth a run.
- **Cyclic-LR SWA tail**: the paper's stronger variant (more diverse minima → flatter average). It is the only SWA
  variant that could in principle exceed cosine-to-0 (the constant-floor family demonstrably cannot), but given
  the constant-floor family approaches 96.22 only from below and both runs maxed out on loss not top-1, the odds
  it clears +0.1pp on top-1 are low. Low-priority.
- **Lookahead optimizer** (weight interpolation that keeps cosine-to-0 intact): a different weight-averaging
  sub-mechanism, but same family the learnings now flag as closed; CIFAR gains typically within noise. Low value.

## Next Steps
1. **Declare the weight-averaging axis CLOSED** (high confidence): EXP-006 (EMA no-op) + EXP-019/020 (SWA
   approaches cosine-to-0 from below, 95.97→96.13) conclusively show trajectory/weight averaging does not beat a
   well-tuned cosine-to-0 schedule on top-1 at this budget. Do not pursue SWA/EMA/Lookahead/cyclic-SWA further.
2. **The cheap per-channel input std-normalization probe** (low confidence): the one remaining untried input-side
   knob (code uses std=(1,1,1)); BN almost certainly absorbs it (expected null) but it definitively closes the
   input-normalization axis for ~one run. Reference exp-report-018/019 Next Steps.
3. **Accept the 96.22 plateau** (high confidence): ~13 axes now exhausted (capacity, block-order, activation,
   attention, weight-decay, more-epochs, aug-policy, aug-strength, label-mixing aug, LR-peak, weight-averaging).
   The model is generalization-bound at fixed k=4 capacity in 300s. Reference project-insights High
   "generalization-bound at fixed capacity" and the new Medium SWA entry.

## Exit Action Results
- No exit actions defined for this goal — skipped.
