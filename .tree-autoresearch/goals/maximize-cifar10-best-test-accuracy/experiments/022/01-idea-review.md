## Prioritized feedback

1. The width-288 proposal must be rebased from EXP-004 to EXP-011 before execution. The old proposal uses EXP-004's 95.40 metric, path counts, live-model evaluation, and no EMA. EXP-022 requires a 95.71% formal threshold, EXP-011 path weights, and expanded full-state EMA checks.
2. Width 288 has strong feasibility evidence but no direct accuracy evidence. EXP-014 measured width 320's systems behavior without querying accuracy, while PyramidalNet studies gradual widening rather than this coarse taper. State the hypothesis as a controlled capacity bet and keep EMA-tail criteria interpretive rather than verdict-changing.
3. Shape-dependent initialization shifts common tensors and the later DataLoader RNG stream. Keep one seed and attribute the result to the complete fixed package; do not copy parent submatrices, burn draws, or rerun.
4. The grouped bottleneck is not executable as specified: geometry, normalization, initialization, MACs, and H20 kernel behavior remain unresolved. Zero terminal projection may also starve inner weights initially.
5. Final-only SE is distinct from EXP-009's four gates, but lacks device timing or evidence that its narrow reach can lift the stable plateau. It returns to the same small-FP32-operation class already found inefficient.
6. Because `best_test_acc` selects a maximum and EXP-011's final-16 range is comparable to the pass margin, record best, final, final-16 mean/range, dose, and EMA samples without changing checkpoint selection, stopping, widths, or seeds.

## Scored verdict

### Fixed 288-Channel Final-Stage Taper
- **Strength of evidence and reasoning: 8.5/10** — exact accounting, a relevant width-320 H20 measurement, multiple-of-32 shapes, no new launches, and explicit RNG/SAM/EMA risks; accuracy benefit remains unmeasured.
- **Potential impact: 8/10** — 18.61% more parameters at 8x8 for an estimated 7-10% throughput cost plausibly offers the needed stable gain while preserving the proven early representation.

### Bottlenecked Grouped Late-Stage Expansion
- **Strength of evidence and reasoning: 4/10** — lacks a fixed architecture, cost model, initialization analysis, and device evidence.
- **Potential impact: 7/10** — extra nonlinear depth has upside, but optimization and launch risks are speculative.

### Final-Block-Only Identity SE
- **Strength of evidence and reasoning: 4.5/10** — literature support and numerical viability do not overcome adverse repo-specific execution evidence.
- **Potential impact: 5.5/10** — narrow reach makes a stable 0.10-point gain less credible than dense late capacity.

## Decisive selection

Select **Fixed 288-Channel Final-Stage Taper**. It is the only candidate combining a direct attack on the diagnosed plateau, exact scope, device-specific feasibility evidence, unchanged kernel topology, and credible effect size. Advance it only with EXP-011-specific thresholds, timing weights, and EMA accounting.
