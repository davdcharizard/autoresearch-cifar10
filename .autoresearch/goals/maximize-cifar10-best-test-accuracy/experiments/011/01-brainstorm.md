# Brainstorm EXP-011
**Created**: 2026-08-06

## Web Search & Literature Review

- **CutMix** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/cutmix.md`): alpha 1.0 is the evidence-backed shape setting; local probability and phase exposure remain the clean tuning levers after EXP-010 validated the method.

## Experimental History Review

- EXP-010 is the new 94.15% baseline: alpha-1 CutMix on 49.77% of N1/M7 plateau batches improved top-1 by 0.60 points, lowered final NLL to 0.1934, and retained 99.10% of EXP-007 steps.
- The strong checkpoint was 89.73%, only 0.35 below EXP-007, so p=0.5 did not cause the predicted compounded-underfit failure. The first weak checkpoint improved by 0.20 and accuracy finished at its best, suggesting useful regional features plus remaining refinement demand.
- EXP-005 showed that switching the complete strong transform off at 75% hurts without CutMix. Any timing candidate should therefore preserve N1/M7 through 80% and isolate CutMix removal, rather than repeat the full early weak switch.
- Alpha changes are less interpretable: alpha below one is U-shaped rather than uniformly weaker, while alpha above one concentrates balanced composites. Probability or CutMix-only duration is the narrower validated search space.

## Collected Ideas

## Combinations

## Candidate Ideas

### Stop CutMix at 70%, Retain N1/M7 to 80%
**Summary**: At 70% elapsed time, explicitly shut down the mixed strong loader and rebuild an N1/M7 loader with default hard-label collation. Keep that transform and `lr=0.1` until the validated 80% boundary, then perform the existing transition to the weak crop/flip tail.

**What it targets**: Provide 10% of the budget for hard-label classifier/BN adaptation while preserving the full validated N1/M7 duration, potentially converting CutMix features earlier without repeating EXP-005's early full weak switch.

**Reasoning**: EXP-010's hard tail improved continuously to termination, so recovery demand may limit the result. This isolates CutMix duration rather than probability and preserves broad strong views through 80%.

**Sources**: EXP-005 and EXP-010 reports; persistent-worker lifecycle learnings.

**Estimated Effort**: medium

**Risk Assessment**: A second loader lifecycle adds wall time and implementation risk, changes the RNG/data stream more broadly, and 70% is not literature-grounded. Hard-label N1/M7 may not approximate weak-tail recovery because its inputs remain difficult.

### Decrease CutMix Probability to 0.25
**Summary**: Change only `CUTMIX_PROBABILITY` from 0.5 to 0.25, preserving every other accepted EXP-010 mechanic.

**What it targets**: The 0.35-point strong-checkpoint deficit and rising endpoint suggest that less composite exposure might preserve more clean strong-phase fit while retaining enough regional supervision.

**Reasoning**: A quarter probability still affects roughly 20% of total updates and keeps the successful compose-not-replace mechanism. It is the cleanest conservative test of whether p=0.5 slightly overshoots the optimal regularization strength.

**Sources**: EXP-010 trajectory and report; `knowledge/papers/cutmix.md`.

**Estimated Effort**: low

**Risk Assessment**: The strong deficit was small and the first weak checkpoint was already better than EXP-007, so reducing CutMix may simply dilute the mechanism that produced the 0.60-point gain.

### Increase CutMix Probability to 0.75
**Summary**: Change only `CUTMIX_PROBABILITY` from 0.5 to 0.75. Preserve alpha 1.0, worker collation/RNG isolation, N1/M7, the 80% transition, width 2, optimizer, and hard weak tail.

**What it targets**: EXP-010 demonstrated regional mixing benefits without strong-phase collapse. More mixed plateau batches may strengthen partial-object and localization invariance enough to clear the new 94.25% threshold.

**Reasoning**: The p=0.5 switch checkpoint remained close to EXP-007 and the tail gained 0.60 points with lower NLL, leaving measured regularization headroom. Hard/soft CE costs were identical, so increased probability should retain exposure.

**Sources**: EXP-010 `04-analysis.md`; `knowledge/papers/cutmix.md`.

**Estimated Effort**: low

**Risk Assessment**: The apparent headroom may be misleading: 75% CutMix can slow clean-boundary fitting, lower the strong checkpoint, and demand more hard-tail recovery than 20% time provides.

## Review

Claude completed the mandatory external review with exit code 0; no fallback reviewer was used. It found p=0.25 contradicted EXP-010's already-better first weak checkpoint, and found the 70% timing option weakly grounded with an extra lifecycle/RNG confound. It selected p=0.75 as the only lever direction supported by successful recovery plus a rising endpoint. I accept its request to preserve the 87.08% switch checkpoint as a pre-registered compounded-underfit diagnostic and to treat a bare 0.10-point pass skeptically. Full critique and scores are in `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's p=0.75 pick. It cleanly strengthens the validated regional-mixing mechanism with a one-literal change, preserves every successful lifecycle detail, and has effectively zero throughput risk based on EXP-010's hard/soft timing parity.

## Chosen Idea
**Selected**: Increase CutMix Probability to 0.75

**Why this idea**:
EXP-010's 0.35-point switch deficit fully recovered into a +0.20 first weak checkpoint and +0.60 final gain, so the evidence does not support reducing mixing. P=0.75 has the highest plausible ceiling without adding a new mechanism or loader transition; the fixed hard tail remains the recovery stage.

**Hypothesis**:
Increasing only `CUTMIX_PROBABILITY` from 0.5 to 0.75 will strengthen regional invariance while retaining near-identical optimizer exposure and a recoverable strong checkpoint, raising `best_test_acc` from 94.15% to at least 94.25%. A switch checkpoint below 87.08% will diagnose excessive compounded regularization if the primary gate is missed.
