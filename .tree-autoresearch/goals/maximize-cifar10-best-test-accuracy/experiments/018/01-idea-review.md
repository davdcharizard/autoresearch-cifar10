# Claude Opus Adversarial Idea Review — EXP-018

## Prioritized Feedback

### Cross-Cutting

1. A local 95.33 pass does not advance the 95.61 global frontier; frame EXP018 as a composability probe and report whether 95.61/95.71 are reached.
2. Max-selected accuracy across roughly 144 evaluations is noisy. Require final-16 mean/range beside `best_test_acc`. Claude proposed hinging the verdict on the mean; the agent rejects that part because the goal's frozen primary metric and necessary condition are explicitly `best_test_acc`.
3. A 3% timing allowance can lose roughly 840 steps, potentially confounding an effect near the formal threshold. Report achieved steps prominently and tighten intervention-specific overhead gates where feasible.

### Direct Canonical Lookahead

- Evaluate the slow weights, or ensure evaluation boundaries coincide with Lookahead synchronization. With 195 parent steps per full epoch and `k=5`, every complete epoch is naturally synchronized; the final partial epoch still needs explicit slow-weight evaluation and exact restore.
- Instrument slow-fast normalized distance so an EMA-redundancy null is interpretable.
- Canonical retained momentum is close to a damping/LR-path change and may be null, but it is a legitimate fixed mechanism.
- Its strongest evidence is in-goal: SAM and EMA both improved this lineage's optimization path, and sparse Lookahead should preserve essentially all exposure.

### Training-Only Companion Classifier

- Deeply-Supervised Nets evidence is era/architecture mismatched: this BN residual network is not demonstrably gradient-starved.
- It partially repeats EXP009's small-kernel latency risk and has two weakly calibrated choices (attachment and weight).
- Mid-level CutMix targets can be especially noisy after spatial pooling, and forcing early linear separability can harm downstream features.
- Any isolated initialization must use a forked/dedicated RNG mechanism that restores global state exactly.

### Clean GC Retry

- The retry is procedurally legitimate and its corrected harness directly addresses EXP017's two defects.
- Ubiquitous BatchNorm makes the removed common-mode convolution direction plausibly redundant; this lowers the expected effect.
- Seventeen reductions have the highest exposure risk. If revisited, tighten latency tolerance and make removed-norm fraction on BN-followed convolutions the main mechanism diagnostic.

## Scores

| Candidate | Evidence / reasoning | Potential impact |
|---|---:|---:|
| Direct Canonical Lookahead | **7/10** — canonical settings plus two same-lineage successes in optimizer geometry/trajectory control; EMA overlap remains unresolved. | **7/10** — matches stable generalization, costs nearly zero steps, and could compose with EMA, though overlap caps the ceiling. |
| Training-Only Companion Classifier | **3/10** — old evidence on gradient-starved networks and two uncalibrated design choices. | **3/10** — plausible representation harm and small-op latency, not merely a null outcome. |
| Clean Reference-Ordered GC Retry | **5/10** — legitimate corrected retry with ECCV evidence, but strong BN-redundancy prior. | **4/10** — likely effect below noise and per-step reductions can erase exposure. |

## Pick

**Direct Canonical Lookahead on EXP002.** It has the strongest same-codebase mechanistic support and the lowest exposure confound. Required refinements are slow-weight evaluation/alignment, final-16 context, explicit achieved-step reporting, and slow-fast distance diagnostics.
