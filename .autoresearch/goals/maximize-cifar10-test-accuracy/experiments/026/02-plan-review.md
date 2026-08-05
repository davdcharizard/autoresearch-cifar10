# Adversarial Plan Review - EXP-026

## Concerns

1. RandAugment's worker RNG draws would otherwise shift later crop/flip draws by roughly the same scale as the target margin, weakening attribution and resembling an augmentation-stream reroll.
2. The plan must use the same multiprocessing context in preflight and production; shared-state correctness cannot rely on ambiguous defaults.
3. The wall projection should combine the historical accepted total with a live absolute candidate-epoch estimate, not only a hardcoded differential model.
4. A fixed `[195,198]` transition-time window can reject a legitimate slow epoch; the actual invariant is an epoch-boundary step lag below `len(train_loader)`.
5. CPU augmentation uses additional wall/CPU work outside the 300 counted seconds; results should be described as equal counted compute, not equal wall time.
6. The preflight should verify the 65% crossing epoch exhausts normally and remains separated from the terminal budget break.

## Disposition

- Adopt concerns 1, 3, 4, 5, and 6. Give RandAugment a lazily cloned per-worker RNG stream that is swapped in only around the torchvision call, restoring the accepted worker RNG in `finally`; require exact accepted crop/flip/tail replay. Use dual conservative wall projections and the step-lag cutoff invariant.
- The host was directly checked with `multiprocessing.get_start_method()` and reports `forkserver`, contrary to concern 2's assertion that the actual default is `fork`. Still adopt its underlying requirement by passing one explicit `multiprocessing.get_context()` object to both the shared `Value` and production DataLoader and exercising that exact context in preflight. A multiprocessing `Value` remains shared when transferred to forkserver workers; the marker test is authoritative.
