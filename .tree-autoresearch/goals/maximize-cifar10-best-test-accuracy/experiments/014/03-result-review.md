# EXP-014 Adversarial Result Review

**Reviewer**: Claude Opus via Claude Code 2.1.220, tools disabled, no fallback model
**Verdict**: PASS

## Blockers

None.

## Independent Recalculation

- All five ratios reconstruct as paired candidate/parent rounds. Sorted ratios are `1.15900855, 1.16063778, 1.16097502, 1.16311842, 1.16352946`; median `1.1609750206` and maximum `1.1635294643` match the raw record.
- The median ratio fails `<=1.15` in every individual round, with no rounding ambiguity. The maximum-ratio `<=1.20` condition passes.
- Parent drift recomputes to `0.00775685`; ratio MAD/median to `0.00169381`; both pass.
- Projected steps `22220.9777`, epochs `113`, EMA samples `137.8152`, and total `520.0007s` all pass their preregistered arithmetic checks.
- The four weights sum exactly to `25798/25798`. Exactly one gate failed, and it was the mandatory stop gate.
- No integrity issue was found: source scope, parent reference, GPU, evaluator guard, correction class, and no-launch behavior were consistent with the plan.

## Nonblocking Concerns Adopted

- The 40 synthetic evaluation forwards are measured but not included in the charged-path weighted ratio; call it a training-step ratio rather than whole-run ratio. Their candidate cost was higher, so inclusion could not reverse the failure.
- The parent microbenchmark weighted step (`12.655 ms` median round) is about 9% slower than `300/25798`, so step/epoch/EMA projections are approximations. They were non-decisive and passed.
- Passing derived exposure/total checks cannot offset the failed median-ratio gate; report them only as arithmetic consistency checks.
- Candidate-only `652.138 MiB` is the binding memory result; `748.484 MiB` joint-process allocation remains informational.
- The conditioning trace and relative SAM values establish only finite/non-degenerate behavior, not an accuracy or capacity mechanism.

## Classification

`crash` with metric `NaN`: this is the schema's mechanical representation of a valid preflight rejection with no primary metric, not a literal runtime crash. `no-improvement` would require a valid measured accuracy; `invalid` would require an integrity violation.

## Recommended Key Learning

Widening WRN-16-4's final stage to 320 costs about 16% weighted step latency on H20, tripping the 1.15 preflight gate before any metric run.
