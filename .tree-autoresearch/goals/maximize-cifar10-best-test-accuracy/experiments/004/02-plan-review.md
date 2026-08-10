# Adversarial Plan Review: EXP-004

**Reviewer**: Claude Code cross-model harness

## Prioritized Concerns

1. **Autocast caching could nullify the perturbation.** If both passes share one autocast context, cached BF16 weight casts may be reused. Require separate autocast contexts and a smoke proving the perturbed second-pass loss or gradients differ from an unperturbed replay.
2. **The near-zero gradient predicate is undefined.** Pin the exact finite/nonzero condition and whether a rejected pulse affects audit counters; an ambiguous late-training abort can waste the one allowed run.
3. **Step/pulse expectations omit fixed SAM overhead.** RNG state copies, norm reduction, snapshots, perturbation, and restoration can cost several seconds. Distinguish soft expected ranges from hard mechanism-underexposure floors.
4. **The ordinary parent path lacks a differential check.** Avoid refactoring its loss path, or explicitly prove fixed-logit clean and CutMix behavior remains identical.
5. **Line-based hang monitoring mismatches carriage-return progress output.** Monitor file size/mtime and normalize carriage returns for compact progress inspection.
6. **The proposed CPU smoke imports a module that constructs the evaluator and still needs CUDA RNG helpers.** Use GPU-0 smokes with the existing local dataset rather than claiming a device-independent CPU test.
7. **The final audit-line fields are inconsistent.** Include first applied step and progress so the final log can support the pre-75% inactivity check.
8. **Failure-stage restoration is underspecified.** Restoration must be guarded by explicit stage flags so failures before snapshot, after perturbation, or during the second pass cannot restore uninitialized state or leave BatchNorm flags changed.
