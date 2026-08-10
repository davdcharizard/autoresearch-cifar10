# External Claude Plan Review - EXP-017

The mandatory external Claude plan review completed successfully. No fallback reviewer was used.

## Prioritized Concerns

1. A 94.25-94.35 single-seed result is noise-consistent weak evidence even though it formally clears the user-defined gate; two fresh projection initializations further separate the trajectory from EXP010.
2. Fork-and-restore construction alone reuses the same RNG substream for a projection and the next shared convolution. Shared-state equality still passes, so the structural gate would not reveal this hidden correlation.
3. Ratio `1.055` does not arithmetically guarantee 25,500 projected steps; the binding threshold is approximately `1.0548`.
4. BN makes the `[0.25,4.0]` shortcut/residual RMS band primarily an implementation tripwire, not the main first-update safety evidence. Update norm, replay loss, and concentration carry that burden.
5. A production run below 25,500 actual steps can formally pass accuracy but only supports an attribution-weak accuracy result.
6. The mechanical-defect repair clause must explicitly forbid rerunning any valid run regardless of an unfavorable accuracy number.

## Adopted Corrections

- Use one dedicated CPU `torch.Generator` derived from the active `torch.initial_seed()` and consume it sequentially for both projection Kaiming initializations. Keep constructors inside RNG forks and skip the projections in model-wide initialization. This avoids global-stream shifts and fork-substream reuse without a tunable secondary seed.
- Tighten the paired timing ratio to `<=1.0548` while retaining the separate `>=25,500` projected-step condition.
- State that 94.25-94.35 is a formal improvement but noise-consistent weak mechanism evidence; larger results carry stronger support.
- Reframe RMS as a structural/catastrophic tripwire and name update/replay/concentration as the load-bearing safety gates.
- Label any accuracy pass below 25,500 actual steps as accuracy-only and attribution-weak.
- Define a valid run independently of accuracy and make every valid run non-rerunnable; repair applies only before a usable valid summary exists.
