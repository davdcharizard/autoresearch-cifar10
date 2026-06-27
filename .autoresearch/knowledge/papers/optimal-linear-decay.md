# Optimal Linear Decay Learning Rate Schedules and Further Refinements

- **Source**: arXiv 2310.07831 (Defazio, Cutkosky, Mehta, et al.)
- **Companion**: "Straight to Zero" (arXiv 2502.15938) — independent LLM-budget replication
- **Consulted**: EXP-016 brainstorm (2026-06-10)

## Key Claims

- Across the most comprehensive schedule evaluation to date (10 diverse problems: vision, LLMs, logistic regression), **warmup + LINEAR decay to zero (lr ∝ 1 − t/T) is the best non-adaptive schedule, outperforming cosine**.
- The linear shape arises from theory (worst-case-optimal for SGD-style methods), not heuristics; warmup itself also emerges from the same theory.
- Cosine's failure mode per the analysis: it over-invests in the just-after-peak plateau and under-anneals through the middle, with a flat near-zero tail.
- Their "refined" (gradient-norm-adapted) schedules go further: rapid annealing near the very end — suggesting follow-ups beyond plain linear if linear wins.

## In-Project Relevance

- Linear and cosine anneals have IDENTICAL LR-time integral (both 0.5·peak over the anneal) — a schedule-FAMILY swap is the unique schedule move that holds the certified-optimal total heat (goal-learnings: heat closed both sides, EXP-010/014/015) constant while changing shape.
- Shape comparisons transfer to our time-keyed budget cleanly (progress ≡ t/T at flat dt), unlike LENGTH intuitions (papers/why-warmup-lr.md caveat, EXP-014).
- In-domain precedent: cifar10-fast used piecewise-linear at our regime (batch 512, one-cycle, mixed precision).
- **EXP-016 result: did NOT transfer — −0.50pp.** The paper optimizes the FINAL-iterate value; our metric is best-over-per-epoch-evals under fixed wall clock. Linear's hot late-middle left the run still climbing at cutoff (best at the final epoch, one near-peak eval), while cosine's flat cold tail manufactures ~10 converged evals to max over. Use this paper for final-value-metric settings only; for max-over-checkpoints metrics, prefer schedules with long cold tails (reports/exp-report-016.md).
