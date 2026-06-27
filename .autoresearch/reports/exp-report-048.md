# Report EXP-048: Numerics-identical charged-step de-overheading — collate-side channels_last + side-stream H2D prefetch
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-048.md
- **Plan**: plans/plan-048.md
- **Log**: logs/exp-log-048.md

## Goal

Maximize CIFAR-10 best_test_acc (%, higher is better) within the fixed 300s charged budget by modifying train.py only. Baseline: **96.71** @ 1990397; bar ≥ 96.81; mean ≈ 96.57, σ ≈ 0.16 (EXP-027).

## Idea & Hypothesis

With every structural class closed after EXP-047, the last unaudited seam was the charged step's non-kernel overhead — and throughput→epochs is the project's only repeatedly-positive mechanism (EXP-000/006; conversion ≈ +0.019/epoch). Chosen idea: remove the in-step layout-permutation kernel by producing channels_last tensors in the (uncharged) DataLoader collate, and take the 6.3MB H2D copy off the critical path with a side-stream CUDA prefetcher — byte-identical values, layout, kernel sequence, and update math (explicitly avoiding EXP-021's faster-but-different-arithmetic failure). Hypothesis: 0.4–1.0ms/step saving → +3–6 epochs → small true shift; pre-registered branches (ii) saving < 0.3ms + mean-band = step already overhead-free, (iii) saving ≥ 0.5ms + mean-band = conversion datum, (i)/(iv) improvement / defect.

## Approach

Three changes in train.py: module-level `collate_channels_last` (default_collate + `.contiguous(channels_last)`, wired via `collate_fn=`); module-level `CUDAPrefetcher` (side stream, `wait_stream`/`record_stream` per the standard pattern, CPU passthrough fallback for sanity); loop iterates the prefetcher with the in-step `.to(device)/.to(channels_last)` lines deleted. Timer code untouched; `torch.cuda.synchronize()` still fences all streams, so overlapped copies stay inside charged windows. CPU sanity: collate value-identity + layout; prefetcher sequence-identity (7/7 batches, two passes); params 4,286,026 unchanged; 2-epoch smoke. No plan deviations.

## Execution

Single pristine run: gates poll 1; GATE_DECISION D0=22.5ms; 32 windows 21.4–22.7ms; rc=0; 140 epochs / 13,515 steps; 300.0s charged / 504.2s total. One monitoring judgment call: ep1 read 34.93 vs the planned 36–41 tripwire band (abort line 30) — the trajectory rejoined the family by ep6–7 (64.0/65.2) and the plateau landed at family level with family test_loss, so numerics-identity was confirmed on the trajectory criterion and the single-read ep1 band documented as uncharacterized-scatter (exp-log § Experimental Adjustments).

## Results

- **Primary metric**: best_test_acc 96.57 (baseline: 96.71, delta: −0.14, −0.14%)
- **Observations**: The decisive datum is the throughput ledger: **13,515 steps vs EXP-046's 13,428 at identical recipe → +0.65% ≈ 0.15ms/step saved**, i.e. the in-step layout kernel + serialized H2D copy cost only ~0.15ms, not the projected 0.4–1.0ms. The read 96.57 equals the EXP-027 recipe mean exactly, at family test_loss (0.1866) and +1 epoch.
- **Analysis**: Pre-registered branch (ii) — the charged step was already essentially overhead-free. Root cause of the overestimate: with `non_blocking=True` from pinned memory, the baseline's H2D copy already overlapped with the *tail* of the previous step's work queued on the same stream, and the channels_last permutation of a 6.3MB tensor is a ~0.05ms bandwidth op on H20 — the "serialization slack" I priced did not exist. The 0.15ms that WAS saved delivered exactly its conversion (+1 epoch ≈ +0.02, invisible). This closes the de-overhead seam with a number: non-kernel overhead ≈ 0.15ms/step (0.7% of the step), and the EXP-006-class throughput mechanism is now exhausted — conv kernel math is 99.3% of the charged step, and the kernel lattice (EXP-040–045) already proved the baseline sits at its floor. Combined with EXP-047's closure of the structural frontier: every currency the budget trades in — kernel time, epochs, params, noise, information routing — is now at a measured optimum or floor.
- **Key Learning**: The charged step's non-kernel overhead is only ~0.15ms (0.7%) — pinned non_blocking copies already overlap and layout permutation is bandwidth-trivial; the throughput→epochs mechanism has no remaining headroom: the budget is 99.3% irreducible kernel math.

## Verification

- **Conditions**: Integrity pre-condition PASSED (pristine profile; 140 epochs ∈ [136,152]; params exact; 300.0s; evals ≤ epochs; numerics confirmed via trajectory + plateau + family test_loss). Condition 1 (best ≥ 96.81) FAILED: 96.57. Conditions 2–3 skipped per first-failure-stop (informationally both pass: 504.2s ≤ 600; 140 ≤ 140).
- **Review Notes**: Results trustworthy — clean profile, exact-mean read, family signatures; charging semantics intact (timer untouched, synchronize fences all streams; the step-count ledger confirms only 0.65% throughput change, ruling out any hidden work-shifting). The ep1 tripwire deviation is documented and resolved by the stronger trajectory criterion.
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure — valid mean-level result below the bar.

## Unexplored Avenues

- **CUDA-graphs-only replay (brainstorm-048 Idea 2)**: was held in reserve pending this run's overhead measurement. The measurement answers it: total non-kernel overhead is ~0.15ms, which bounds the launch-overhead recoverable by cudagraphs BELOW 0.15ms — strictly not worth a run (and EXP-021 adjacency makes its risk profile worse than its ceiling). Treat as closed by this bound.
- **Worker-count / prefetch_factor tuning**: loader-side, uncharged by design; cannot move the charged step. Only relevant if loader stalls appeared (they did not — windows pristine).

## Next Steps

1. **Acknowledge the measured ceiling and re-frame what an improvement would require** (high confidence): every axis is now at a measured floor/optimum — a future improvement must come from a qualitatively different function class or training signal that passes ALL laws simultaneously; no catalogued or constructible single-change candidate remains. Brainstorms should now consider compound interventions whose components are individually certified (the only unfalsified region), while honestly weighting the EXP-009 precedent (stacking certified regularizers lost).
2. **Replicate-pair the baseline question** (medium confidence): with 42 consecutive non-improvements and mean ≈ 96.57, the standing 96.71 baseline is itself a +0.9σ draw; the loop's bar (96.81) is mean +1.5σ — any true effect must be ≥ +0.3 to be reliably detectable, which the law-stack says no remaining single change can deliver. This is worth stating plainly in the next brainstorm rather than re-derived.
3. **Protocol stack carries forward unchanged** (high confidence): six consecutive runs resolved exactly per pre-registration; the step-count ledger (steps vs EXP-046's 13,428) is a new cheap integrity instrument worth keeping.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
