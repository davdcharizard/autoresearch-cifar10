# EXP-019 Independent Audit

**Experiment**: Corrected reference-ordered Gradient Centralization on EXP-002 (branch `tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-019`, base commit a36dc09)

## Metric decision (recomputed)

- Parent EXP-002 metric in `04-results.tsv`: **95.23** (row 002, commit a36dc09 — matches this branch's base). Threshold = 95.23 + 0.10 = **95.33**.
- `run.log:303` → `best_test_acc: 95.07%`. 95.07 < 95.33 (−0.26 vs threshold, **−0.16 vs parent**). The running `best:` column peaks at 95.07 (epoch 141) and no eval exceeds it, so 95.07 is the true max over all 144 evaluations.
- **Verdict recomputation: no-improvement — matches the recorded FAIL.** The 03-execute deltas (−0.16, −0.26) are arithmetically correct.

## Scope and freshness

- `git diff --name-only a36dc09` → only `train.py`; `git status` shows only `M train.py`, no untracked helper artifacts in the repo. **PASS.**
- Timeline is coherent and same-day (2026-08-06): `train.py` mtime 15:23:41 → smoke 15:25 → preflight 15:29–15:30 → run start 15:31:19 → `run.log` mtime 15:38:52 (claimed end 15:38:49). The log was produced by the code as it currently sits. **Not stale.**

## Frozen budget and evaluation

- `training_seconds: 300.0` ∈ [299.5, 301.0]; `total_seconds: 449.8` < 600; `startup_seconds: 1.3`. Exactly 144 eval lines for 144 epochs — one eval per epoch. **PASS.**
- Diff inspection: the eval path, `best_acc = max` selection, eval cadence, CutMix/drop-path/LR config, and `cutmix_seed=42` are untouched. The only eval-adjacent addition is `eval_accuracies.append(test_acc)` of the already-computed value. Weight decay moved from SGD-internal (1e-4) to an external `_foreach_add_` at the same 1e-4 with optimizer `weight_decay=0.0` — equivalent coupled L2, neither omitted nor doubled. GC executes between `backward()` and `optimizer.step()`, inside the `t0` → `cuda.synchronize()` → `dt` region, so all mechanism cost is charged. **No reward-hacking or budget-shifting vector found.**

## GC reconciliation (all recomputed from raw log)

- **Inventory**: 17 tensors (16 conv + 1 classifier), 2,745,264 eligible + 3,626 excluded = 2,748,890 = `num_params`; 2,266 rows; 27 excluded tensors. Hard-asserted in code and printed. ✓
- **Cadence**: 55 audits = 1 (step 1) + ⌊27,976/512⌋ = 54. ✓
- **Path decomposition**: 10,261 + 10,416 + 7,299 = 27,976 = `gc_calls` = `num_steps`; cutmix path = applied batches (10,261), early-clean = 20,677 − 10,261, late-clean = 27,976 − 20,677. ✓
- **Dose arithmetic**: 475,592 = 17 × 27,976; 76,801,505,664 = 2,745,264 × 27,976; 63,393,616 = 2,266 × 27,976. ✓
- **Energy**: removed/regularized = 6.41621/29.10161 = 0.220476 (fraction), √ = 0.469549 (norm ratio) ✓; conv 0.418894 ✓; classifier 0.932131 ✓; conv + classifier components sum exactly to totals ✓; decomposition error 4.0e-9 ≤ 1e-5 ✓; residual 3.5e-9 ≤ 1e-6 ✓; zero nonfinite; integrity `PASS` with a hard `RuntimeError` guard that would have failed the exit code otherwise.
- **Tail-16**: all 16 values match raw eval lines for epochs 129–144 (spot-checked including the truncated middle: ep129=94.91, ep130=94.82, ep131=94.99); mean recomputes to exactly 94.966875; min/max/final/premium (0.103125) all correct.
- **Preflight**: `/tmp/exp019_preflight.log` still exists and its JSON matches 03-execute exactly (median ratio 1.007687, max 1.014301, MAD/median 0.006098, drift 0.004559, projected 27,736 steps/143 epochs, 655,360-byte allocation growth, zero evaluator calls). Actual run (27,976 steps/144 epochs) exceeded projections.

## Corrections (minor, none affecting the verdict)

1. 03-execute cites the summary as `run.log` L294–L312; the summary block is L295–L312 (L294 is the epoch-144 eval line). Cosmetic citation offset.
2. `/tmp` contains stale identically-prefixed artifacts from June 30 (`exp019_smoke.py`, `exp019_run.sh`, etc.) belonging to an unrelated earlier session; the operative Aug-6 files are distinguishable by mtime, but the prefix collision is a hygiene risk for future audits.
3. Preflight JSON shows `timing_candidate_gc_calls=1095` (timed rounds + warmup) alongside the 1,056-step allocation trace; 03-execute mentions only the latter — internally consistent, just incompletely narrated.

## Trustworthiness of the no-improvement result

The 95.07 result is trustworthy: dose was full (27,976 steps / 144 epochs vs parent 27,950/144, CutMix ratio 0.4963 vs configured 0.5), budget was exactly honored, the mechanism demonstrably executed at scale (22% of regularized gradient energy removed — a large intervention, ruling out the "GC was a no-op" alternative), all integrity assertions passed with exit 0, and the tail-16 plateau (mean 94.97, max 95.07) shows a stable converged run rather than an unlucky single eval. The negative result is genuine evidence against this exact GC composition, consistent with the preregistered decision rule.

AUDIT_VERDICT: PASS
