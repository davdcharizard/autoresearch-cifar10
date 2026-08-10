# Claude Adversarial Plan Review — EXP-014

- **Reviewer**: Claude Opus, read-only tools, no fallback model
- **Initial result**: BLOCK pending resolution

## Prioritized Concerns

1. Architecture shape changes alter model initialization consumption and therefore the later global CPU RNG state used by DataLoader workers. A bare `+0.10` maximum can be confounded by the fixed-seed package realization; the plan must name this alternative explanation and keep the higher tail bar prominent.
2. The preflight memory gate did not reset CUDA peak statistics or define candidate-only ownership in the joint process.
3. Importing parent/candidate instantiates test loaders, while the 40 evaluation forwards did not mechanically forbid calls to the test evaluator. Use synthetic/train tensors and monkeypatch evaluator calls to raise.
4. Numeric stability-gate failures could be relabeled as harness errors and rerun. Define a closed harness-error class and make any emitted numeric gate decisive.
5. Several projected dose/runtime checks are implied by the stricter median-ratio gate, while realized dose remains scientific rather than formal. Label derived checks honestly and preserve the frozen formal verdict.
6. `p90` over five round ratios was undefined by population/interpolation. Replace it with an explicit maximum-round-ratio gate.
7. A hardcoded MAC literal in production code would self-verify and exceed the minimal architecture change. Keep MAC reconciliation in the external harness/artifact.
8. Diagnostic placement was underspecified. Reuse the existing debiased scalar only after the loop and add evaluation progress only outside the charged step; forbid added operations between `t0` and `dt`.
9. Six drop-path draws occur only when early `drop_scale>0`; at terminal `drop_scale=0`, the correct count is zero.
10. The two-decimal metric threshold needs an exact basis. CIFAR-10 has 10,000 test examples, so `95.71%` corresponds to exactly 9,571 correct; verify that granularity or print more precision.
11. The tree command path was abbreviated, and ordinary `git status` cannot inspect ignored artifacts. Use the exact shared-script path and rely on tracked diff scope; keep all harnesses in `/tmp`.
12. Name the bare config-line parameter parse explicitly; acknowledge that the lower charged-time bound is structural while the upper bound detects excessive final-step overshoot.

## Resolution Status

- **Final verdict**: PASS
- Claude confirmed that architecture-induced RNG/data realization is an attribution limit rather than seed rerolling: seed 42, fixed width 320, one data realization, and one metric run are locked with no alternate stream or post-result selection.
- It verified all twelve concerns were resolved: candidate-only memory process, evaluator monkeypatch, closed retry class, explicit maximum-round gate, external MAC accounting, outside-charged diagnostics, early/late drop-path assertions, exact metric granularity, executable tree/log checks, and separately stated formal/scientific verdicts.
- Independent arithmetic reconciliation passed for parameters, MACs, path weights, and all ratio-derived projections.
