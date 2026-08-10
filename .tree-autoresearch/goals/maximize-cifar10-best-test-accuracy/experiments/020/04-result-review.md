# EXP020 Independent Audit

**Bottom line: the run is genuine, complete, and correctly reported. The experiment itself is a formal no-improvement (95.24% vs required 95.33%), and the transcript records exactly that.**

## Primary verdict recomputation
- `run.log` L307: `best_test_acc: 95.24%`. Tree store `04-results.tsv` confirms parent EXP002 (commit `a36dc09`) = **95.23**. Delta = +0.01, below the +0.10 gate → threshold 95.33% not met. **Formal no-improvement — matches the transcript's "failed" outcome.** 95.24 > 95.23, so the transcript's refusal to trigger the "≤95.23 closes official-order GC" rule is also correct (preregistered "unresolved positive movement" band 95.24–95.32).
- 95.24 is consistent with the eval stream: max of all 145 `eval ep` lines, first reached at epoch 143.

## Freshness and scope
- `run.log` mtime 2026-08-06 16:14:51 UTC, checked at 16:18 UTC — fresh today. Started 16:07:05 + 461.1 total seconds = 16:14:46, matching the last-write time; the transcript's "Ended 16:15:03" is the outer command return (interpreter teardown), a benign 12 s gap.
- `git diff --name-only a36dc09` → only `train.py`; `git status` shows only `M train.py`, no untracked helpers. The diff (301 insertions) adds only GC machinery, report-only audits, and summary lines; no changes to the evaluator import (`Eval` from read-only `prepare.py`), seeds (42/42, cutmix_seed 42), `TIME_BUDGET_S`, best-accuracy selection (`test_acc > best_acc`), or timing accounting. GC and audit work sit inside the `t0 → cuda.synchronize → dt` window, so mechanism cost is charged to the budget. The optimizer `weight_decay=1e-4 → 0` move is exactly the declared external coupled-L2 placement applied to all 44 gradients. **No reward hacking or evaluator changes.**

## Completeness, evaluations, budget
- Final summary complete: all 10 inherited keys present (L307–L316), exit without traceback, integrity `PASS` printed before the summary.
- Evaluations: 145 `eval ep` lines for `num_epochs: 145` — exactly one per epoch (epoch 145 is a legal budget-exhausted partial epoch with its eval).
- Budget: `training_seconds: 300.0` ∈ [299.5, 301.0]; `total_seconds: 461.1` < 600 s timeout; `startup_seconds: 1.1`. Exposure 28,090 steps / 145 epochs actually exceeds parent's 27,950 / 144 — exposure preserved.

## Arithmetic recomputation (all exact)
- **Inventory, independently derived from the architecture**: 16 Conv2d weights = 2,742,704 elements (432 + 47,104 + 73,728 + 229,376 + 294,912 + 917,504 + 1,179,648), 2,256 rows; excluded = 26 BN tensors (1,808 ch × 2 = 3,616) + Linear w/b (2,570) = 28 tensors / 6,186 elements; totals 44 / 2,748,890. All match the log.
- **Calls/paths**: gc_calls 28,090 = num_steps; 10,292 + 10,453 + 7,345 = 28,090; cutmix applied 10,292 = cutmix-phase steps; eligible 20,745 = 10,292 + 10,453. Dose products: 16×28,090 = 449,440; 2,742,704×28,090 = 77,042,555,360; 2,256×28,090 = 63,371,040. ✓
- **Audits**: 1 + ⌊28,089/512⌋ = 55 samples; phase split 19+22+14 = 55. ✓
- **Energy**: centralized 20.18229941021 + removed 4.262522861646 = 24.444822272 vs regularized 24.44482219761 → rel. error 3.04e-9 (matches logged 3.037e-9, ≤1e-5). Norm ratios recomputed: total √(4.2625/24.4448) = 0.41758; phases 0.40711 / 0.42751 / 0.39802 — all match. Phase sums reconcile to 1.3e-15. Components 4.262502784533 + 1.18801e-5 + 8.19702e-6 = 4.262522861646 = removed exactly (scaled error 1.04e-15, PASS). Transcript's "L2 common-mode negligible / raw-order numerically near-identical" claim is supported and properly disclaimed as non-causal.
- **Tail stats**: the 16 logged values match eval lines ep 130–145 verbatim; sum 1521.94/16 = **95.12125** mean, min 95.03, max 95.24, final 95.19, best premium 95.24 − 95.12125 = **0.11875**. All match log and transcript.
- **Phase claim check**: "no uniquely excessive late-phase projection" — supported; late-clean ratio (39.80%) is the *lowest* of the three phases.

## Transcript fidelity
Every citation in `03-execute.md` (L297–L316 line numbers, 145-eval count, energies, 41.7580%, integrity PASS, VRAM 1,180.0 MiB ≈ expected ~1,178.9) matches the raw log exactly. Residual 3.73e-9 ≤ 1e-6, decomposition 3.04e-9 ≤ 1e-5, zero nonfinite — all hard integrity gates genuinely pass in-code (integrity failure would have raised before the summary printed).

No discrepancies found beyond the immaterial 12-second end-timestamp gap.

AUDIT_VERDICT: PASS
