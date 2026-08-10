# Adversarial Plan Review — EXP-015

**Reviewer**: External Claude plan critic
**Status**: completed successfully; no fallback reviewer used

## Prioritized Concerns

1. **Production gamma logging contradicted the selected proposal's scope.** `idea-03.md` forbids logging diffs, while the draft plan required per-evaluation/final gamma RMS/max fields. Either amend the selected method explicitly or remove this instrumentation and its dependent verification.
2. **The launch did not explicitly pin the confirmed H20.** The proposal used `CUDA_VISIBLE_DEVICES=0`; the plan should pin the already-verified sole idle device so `torch.device("cuda")` cannot resolve elsewhere.
3. **Exposure/evaluation diagnostics had ambiguous verdict precedence.** The goal's necessary conditions are accuracy, completion, and time. A small benign step-floor miss should weaken compute-neutral attribution rather than invalidate a genuine pass; more than 19 evaluations does invalidate fairness by adding metric opportunities.
4. **First-update thresholds needed a stated basis.** Gamma 0.25, replay loss 2x, 95% concentration, and 64-step loss 1.5x can veto the run. Explain them as deliberately loose catastrophic-failure sentinels grounded against EXP-014, not accuracy-tuning gates.
5. **A bare 94.25-94.35 pass is weak max-over-checkpoints evidence.** Record it as formal but low-confidence under the user-defined threshold, and require bounded first-update plus fit diagnostics before making a genuine-mechanism claim.
6. **Constructor signature differed cosmetically from the proposal example.** Pick positional-or-keyword or literal keyword-only deliberately; `_make_layer` already calls it by keyword.

## Response

- Accepted concern 1: removed all production gamma logging and all dependent parse requirements. Recruitment remains a strict preflight result; production remains only the selected initialization diff.
- Accepted concern 2: launch with `CUDA_VISIBLE_DEVICES=0` after verifying index 0 is the only visible idle H20.
- Accepted concern 3: `>=26,629` steps and 18-19 evaluations are expected mechanism diagnostics. A step-floor or low-evaluation miss weakens attribution but does not override a formal accuracy/completion/time pass. More than 19 evaluations is invalid because it gives extra chances to maximize the metric.
- Accepted concern 4: retained the proposal's safety gates and documented their loose basis. EXP-014 produced a 1.221 branch/accepted ratio, 9.92x same-batch loss jump, and 100% one-class prediction after one step; the new limits bound gamma to 0.25, loss to 2x, and concentration below 95%. The 64-step 1.5x gate only catches gross sustained fit failure.
- Accepted concern 5: a bare pass is reported as formal low-confidence improvement. Bounded preflight recruitment and a healthy switch support the mechanism, but diagnostics cannot rewrite the user-defined verdict.
- Concern 6 requires no change: literal keyword-only form is a deliberate smaller API surface; callers pass the flag by name.
