# Report EXP-034: Batch 512 With Fully Scaled LR
- **Created**: 2026-07-26

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` from the accepted 94.32% baseline to at least 94.42% within the fixed 300-second counted-training budget and 600-second wall limit. This experiment tested whether batch 512 could use the H20 more efficiently while a fully doubled LR curve preserved useful large-batch optimization.

## Idea & Hypothesis

Change exactly batch size 256 to 512, peak/floor LR `0.2/0.002` to `0.4/0.004`, and the image-equivalent step cap 64,000 to 32,000. The hypothesis required at least 1.10x measured complete-body image rate, 146.308096 projected passes, and 14,287 projected updates before one score. This strict gate was necessary because the operating point roughly halves optimizer, BN, and batch-shared mixup decisions and local exposure-only treatments have repeatedly regressed.

## Approach

The production diff contained only the four constants. An ignored verifier used an independent accepted-source oracle, guarded all evaluator/test access, proved initial model/construction RNG and optimizer-group semantics, exact doubled LR values, the image-equivalent cap, finite FP32 mixup/hard updates, and batch-512 worker cutoff behavior. Its instrumented forkserver arms fully consumed an active epoch and matched 49,664 per-sample worker/index/crop/flip/target traces plus 49,664 exact clean-tail images.

Balanced H20 timing then compared accepted batch 256 and candidate batch 512 for complete early-mixup and hard-label production bodies with reset model/optimizer/RNG fixtures. The protocol printed raw windows before assertions and would publish a provenance-bound timing payload only after all gates passed.

## Execution

The first semantic harness run compared class objects from separate module namespaces and failed even though their source matched. The ignored check was corrected to compare ordered class names alongside the existing source audit; production did not change. Semantic qualification then passed with 987,098 parameters, 97 batches / 49,664 images, and 1,984.41 MiB peak allocation.

The sole throughput measurement was stable but failed the material-gain gate. All four CVs were between 0.0869% and 0.7125%. Per the plan, timing was not repeated, no passing JSON was published, contemporaneous loader timing was skipped, and no score or `run.log` was created.

## Results

- **Primary metric**: NaN (baseline: 94.32%; delta: N/A)
- **Observations**: Accepted mixup/hard medians were 11.6676/11.2630 ms; batch-512 medians were 21.8616/21.4654 ms. Weighted image rate rose from 22,217.0 to 23,571.4 images/s, a 1.06096x gain. This projected 141.11544 passes and 13,780.81 updates, below the required 1.10x, 146.308096 passes, and 14,287 updates.
- **Analysis**: The systems premise worked only weakly. Batch 512 fit easily and increased full-body image throughput by 6.10%, but its steps cost 1.87-1.91x accepted, so it surrendered about 47% of update/BN/mixup decisions for only eight additional projected passes. Given multiple local regressions at even larger exposure gains, this did not justify spending the sole accuracy score. The result closes exact batch 512 with the `0.4 -> 0.004` curve and 32,000 cap; it supplies no accuracy evidence about large-batch training in general.
- **Key Learning**: H20 memory headroom does not translate into material batch scaling here; batch 512 buys only 6.1% image rate while nearly halving decisions.

## Verification

- **Conditions**: Static and semantic gates passed; timing stability passed; the fixed material image-rate/pass/update gate failed; loader and score stages were skipped.
- **Review Notes**: The failure is trustworthy and not infrastructure: raw balanced windows were finite and highly stable, one idle H20 was used, exact scope and worker semantics passed, and no stale payload or scored log existed.
- **Verdict**: crash
- **Verdict Basis**: No primary metric was produced because the preregistered pre-score throughput gate failed.

## Unexplored Avenues

- Batch 384 or 1024 could have different kernel scaling, but selecting an adjacent size after this result would be an unjustified rescue without new optimizer or kernel evidence.
- A different large-batch optimizer could preserve decision-domain dynamics, but that is a new compound mechanism rather than a repair to this operating point.

## Next Steps

- **Low confidence**: Close the remaining batch-shared mixup-strength bracket with alpha 0.1 as a clean one-line experiment despite its negative local prior.
- **Low confidence**: Develop a new low-cost classifier-boundary mechanism such as normalized logits only after deriving an exact optimization and throughput contract.
- **Low confidence**: Revisit architecture only through a new interaction hypothesis that preserves full high-resolution gradients and the accepted deeper-plus-augmentation recipe.
