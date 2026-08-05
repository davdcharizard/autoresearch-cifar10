1. **[Run validity, `02-plan.md:14`, `02-plan.md:71-75`, `02-plan.md:126-127`]** The plan says the first run is “partly a calibration” and includes loop-2 LR retunes. That conflicts with the one fixed-seed/no-sweeping constraint. A bad LR, early divergence, or below-baseline result must count as the experiment outcome, not become a discarded calibration run.

2. **[Throughput gate, `02-plan.md:118`]** If `num_epochs < 142`, the plan labels the run “throughput-confounded” rather than a Muon verdict. But Muon’s extra Newton-Schulz work is part of the method; fewer epochs caused by optimizer overhead is a real failure under the 300s budget, not an external confound unless independently proven.

3. **[EMA correctness, `02-plan.md:40`, `train.py:304-309`, `train.py:343-345`]** Weight renorm is applied before the Muon update, then EMA is updated after the post-update weights. The scored `ema_model` therefore averages weights that are not actually norm-pinned, plus BN buffers averaged separately. The plan’s claim that EMA is optimizer-agnostic misses a concrete scale/BN-buffer mismatch risk.

4. **[Schedule/logging edit, `02-plan.md:50-53`, `train.py:328-330`]** Replacing `lr` with `frac` can leave the existing progress print referencing an undefined `lr`. If implementation does not preserve `lr` or introduce `lr_sgd`/`lr_muon` for logging, the run will crash at step 50.

5. **[Attribution, `02-plan.md:10`, `02-plan.md:14`, `02-plan.md:57-61`]** This is not a single “conv SGD → Muon” change. It also removes conv weight decay and adds per-step norm projection. A pass cannot be attributed cleanly to Muon versus no conv L2 versus renorm/effective-LR changes.

6. **[Chosen-idea drift, `01-brainstorm.md:97-101` vs `02-plan.md:6-12`]** The chosen idea was Muon on conv+fc with decoupled WD and ~0.02 LR/5 NS steps. The concrete plan changes to conv-only, no Muon WD, weight-renorm, LR 0.24, 3 NS steps. The deviations are large enough that the experiment no longer directly tests the brainstorm hypothesis.

7. **[LR justification, `02-plan.md:11`, `02-plan.md:58-60`]** The 0.24 LR transfer is overclaimed. The cited Airbench script uses a different architecture, batch size, schedule length, momentum 0.6, no EMA, and compiled optimizer path. “NS normalizes update scale” does not make momentum/schedule/model transfer automatic.

8. **[Smoke test gap, `02-plan.md:96-103`]** The smoke test prints the post-step norm but does not assert it. A broken or omitted renorm could still print `SMOKE_PASS`. It also runs random tensors on CPU by default, so it does not verify the actual CUDA bf16 path or step-time overhead.

9. **[Smoke test inconsistency, `02-plan.md:22`, `02-plan.md:75`, `02-plan.md:93-95`]** The milestone says lower singular values on random inputs are expected and only max `<2` matters, but abort criteria mention singular values outside `[0.5,1.5]`. That creates an ambiguous false-fail gate.

10. **[Metric genuineness, `02-plan.md:112-119`, `train.py:349-373`]** Verification trusts summary `best_test_acc` but does not require it to equal the max per-epoch evaluator trace. Since only `train.py` is editable, this leaves a reward-hacking path via summary/best bookkeeping.

11. **[Budget verification, `02-plan.md:105-110`]** Necessary condition 1 greps `training_seconds` but does not make `training_seconds≈300` a hard pass/fail check. A timer bug or premature stop could still pass if a summary metric prints.

12. **[Scope verification, `02-plan.md:116-117`]** Checking only `prepare.py` and top-level `*.py` does not prove “only `train.py` changed.” It misses staged changes, non-Python files, and other benchmark-affecting files.
