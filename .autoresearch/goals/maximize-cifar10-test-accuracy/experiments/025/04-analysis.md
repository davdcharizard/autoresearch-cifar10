# Report EXP-025: Diagnostic-Free Full Two-Gate SE Closure
- **Created**: 2026-07-26

## Goal

Maximize fixed-budget CIFAR-10 `best_test_acc` above the accepted 94.07% baseline, with 94.17% required for acceptance. This experiment asked whether removing read-only diagnostics from EXP-017's 94.16% full-SE treatment could recover enough exposure to justify one fixed-seed score.

## Idea & Hypothesis

Recreate EXP-017's exact two ratio-16, globally conditioned stage-3 SE gates at seed 17017 while removing every training-time diagnostic buffer and reduction. The hypothesis required exact treatment semantics, at least 137 projected data passes, and then a single score of at least 94.17%; a timing miss prohibited scoring or adjustment.

## Approach

`train.py` gained an opt-in `Stage3SE` path on both stage-3 residual branches. Gates were attached after accepted whole-model initialization inside a restored CPU RNG fork, used the archived seed/init oracle, scaled only the residual before shortcut addition, and contained no observation state or work. An ignored preflight instantiated the real production constructor path and checked accepted common state, CPU/CUDA RNG, logits, placement, initialization, gradients, optimizer grouping, parameter count, and matched H20 throughput.

## Execution

One semantic preflight and one preregistered throughput preflight ran on the local NVIDIA H20. Semantic verification passed completely. Three balanced timing windows per mixup/hard regime had CVs below 0.5%, but weighted timing projected only 136.900785 passes, below the binding 137.0 floor. Execution stopped immediately; no scored command, retry, threshold change, or `run.log` occurred.

## Results

- **Primary metric**: NaN (baseline: 94.07%, delta: N/A; scored run not launched)
- **Observations**: The exact diagnostic-free gates retained 96.4769% of accepted weighted throughput. Accepted/candidate weighted step times were 13.094159/13.572319 ms; projected exposure was 136.900785 passes. Semantic checks passed at 696,042 parameters.
- **Analysis**: Removing diagnostics recovered much of EXP-017's lost work, but not enough to enter the preregistered exposure regime. Stable, low-CV timing makes an infrastructure explanation implausible. The remaining overhead belongs to the full dense conditional gate function itself, so the exposure-only rescue cannot be tested honestly without lowering a fixed threshold after seeing the result.
- **Key Learning**: Diagnostic-free full SE still projects only 136.90 passes; dense cross-channel conditioning, not observation alone, keeps this near-miss outside its exposure regime.

## Verification

- **Conditions**: Baseline, device/scope/compile, and semantic identity passed; projected exposure failed at 136.900785 < 137.0; scored-run conditions were skipped.
- **Review Notes**: Timing CVs ranged from 0.000466 to 0.004804, and accepted/candidate paths used balanced order plus private RNG streams. The failure reflects the preregistered feasibility boundary rather than parsing, stale output, or infrastructure error.
- **Verdict**: crash
- **Verdict Basis**: No scored metric was produced because the candidate failed the mandatory pre-score throughput condition.

## Unexplored Avenues

- None for the exact full two-gate exposure closure: lowering the gate, changing ratio/placement/seed, or rerunning would violate the preregistered isolation. A new attention function class would be a separate hypothesis, and prior simplifications already lost the positive signal.

## Next Steps

- **Early-only mild RandAugment (medium confidence)**: test an orthogonal input-invariance lever while preserving the validated mixup and hard-label tail; implement phase control without persistent-worker leakage.
- **Weaker alpha-0.1 mixup (low confidence)**: finish the remaining isolated strength-side bracket only if no stronger orthogonal augmentation plan survives review.
- **Neighbor-mixing conditional gates (low confidence)**: retain only as mechanism mapping; unordered local channel mixing is unlikely to reproduce dense global interaction.
