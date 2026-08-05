# Proposal idea-01: Cosine one-cycle decay (vs the linear triangular shape)

## Core change (train.py only)
The current LR schedule (train.py:286-290) is a **linear triangular** one-cycle: ramp 0→PEAK over the first `PCT_START`(=0.15) of the *training-time* budget, then **linear** decay PEAK→0. This linear-decay shape was set in EXP-001 and never tuned. Add a `SCHEDULE` env that changes ONLY the post-warmup decay shape (warmup, peak, EMA/TTA gates all unchanged):
- `tri` (default = current linear decay, exact baseline / regression control).
- `cos` (PRIMARY): cosine decay after warmup — `lr = PEAK_LR * 0.5*(1 + cos(pi*q))` where `q = (progress - PCT_START)/(1 - PCT_START)` ∈ [0,1]. Finishes at **exactly 0** (full anneal preserved — the EXP-019 review's "finish at 0" requirement).

```python
import math
progress = min(1.0, total_training_time / TIME_BUDGET_S)
if progress < PCT_START:
    lr = PEAK_LR * progress / PCT_START
else:
    q = (progress - PCT_START) / (1.0 - PCT_START)
    if SCHEDULE == "cos":
        lr = PEAK_LR * 0.5 * (1.0 + math.cos(math.pi * q))
    else:  # tri
        lr = PEAK_LR * (1.0 - q)
```
`SCHEDULE` env (`tri`/`cos`). Exactly throughput-free — a per-step scalar formula change; num_epochs identical to baseline.

## Mechanism — why this is a DIFFERENT, throughput-free lever
This changes WHERE in training the optimizer spends its steps, NOT the model, per-step compute, or regularization. Cosine holds the LR HIGHER for longer in the early-mid phase then drops STEEPLY at the end, vs linear's constant-rate descent — a different exploration/anneal balance that selects a different (plausibly flatter/better-generalizing) minimum. EXP-001 established that **most accuracy lands in the low-LR tail of a completing one-cycle** (project-insights Medium), so the decay SHAPE that governs that tail is a direct, untried handle on the final minimum.

## Why it targets the limiter
The limiter is a budget-limited **generalization ceiling** (project-insights High, EXP-014; 14 nulls). Unlike capacity/optimizer/aug/BN-noise/downsampling/attention (all saturated or under-anneal-trapped), the decay shape is **throughput-free** (zero under-anneal risk — the #1 failure mode behind 5+ nulls) and was EXPLICITLY flagged by EXP-012's analysis as "an untried lever with ceiling clearly above noise." It directly governs the anneal-tail generalization that dominates final accuracy.

## External evidence (NEW this loop — the strongest for any cheap lever on this goal)
- **MosaicML LR-schedule benchmark**: cyclic/one-cycle schedules "do not necessarily lead to improved accuracy when compared to cosine decay … in many instances the cyclic tradeoff curve underestimated the standard [cosine] tradeoff curve by a margin of 0.5% validation accuracy," and "results held for CIFAR-10 as well." Our schedule is the cyclic/linear kind → cosine could plausibly be ~0.5% better (https://cameronrwolfe.substack.com/p/the-best-learning-rate-schedules).
- **fastai one-cycle uses COSINE annealing for the curve shape**, not linear — our EXP-001 linear-decay one-cycle is the less-standard variant (https://docs.fast.ai).
- **SGDR / Bag-of-Tricks** (He et al. CVPR 2019, arXiv:1812.01187; Loshchilov arXiv:1608.03983): cosine is the standard research-grade CIFAR schedule; cosine decay smoothly to 0 is a recognized accuracy lever.

## Throughput
Exactly neutral — a scalar arithmetic change. num_epochs must stay ~150 (verify; any drop indicates a bug, not the schedule). No fused-kernel/op concerns. This sidesteps the under-anneal trap entirely.

## Design — SAME-SESSION multi-cell (verdict keyed on cA)
- c0: `SCHEDULE=tri` — full-speed same-session anchor AND a regression check (LR trace bit-equivalent to the current formula at sampled progress points).
- cA (PRIMARY): `SCHEDULE=cos` — cosine decay. Determines the verdict.
- cB (diagnostic): a second cosine operating point, e.g. cosine with a marginally shorter warmup (PCT_START 0.15→0.10, since cosine is gentler early) — informational only, isolates warmup sensitivity; does NOT bear on the verdict (avoids schedule-search on the test metric).

## Correctness / EMA / eval
- Only the LR scalar changes. EMA warmup gate (`progress>=0.15`) and flip-TTA gate (`progress>=0.8`) key on `progress`, NOT on LR, so they are untouched. Model/eval path byte-identical. No new params, no VRAM change, no dtype concerns.
- `SCHEDULE=tri` must reproduce the current baseline LR trace exactly (regression smoke at sampled progress).
- Smoke: print the LR trace for tri vs cos — both ramp identically 0→PEAK over PCT_START, both monotone-decay after, cos hits exactly 0 at progress=1, tri matches the old formula.

## Verification (carry the EXP-019 hardened protocol)
- cA(cos) ≥ 96.48 AND cA > same-session c0 by >0.1pp, replicated with a mandatory confirmation re-run on any apparent win.
- num_epochs ≈ 150 (throughput-free premise — must stay full); cos changes the early-LR so ep25 WILL differ from c0 (judge full-anneal best≈final, not ep25 parity); fully annealed (LR→0).
- Integrity: train.py-only; prepare.py byte-unchanged; ≤1 eval/epoch; seed 42; summary best == per-epoch max; `tri` ≡ baseline smoke. Background nvidia-smi contention sampling.
- ON A WIN: bake `SCHEDULE=cos` as default.

## Hypothesis
Cosine decay (vs the linear triangular decay) selects a better-generalizing minimum in the low-LR tail and lifts best_test_acc ≥96.48 over the same-session control by a clear >0.1pp margin at ~150 epochs (throughput-free), replicated on a confirmation re-run. If it ties, the linear one-cycle is already near-optimal for this net/budget and the anneal-shape lever is exhausted — strengthening the genuine-ceiling diagnosis and pointing to a wholesale different backbone.

## Effort: low. Risk: (1) a well-tuned triangular one-cycle is already strong; the MosaicML ~0.5% is on ResNet-50/longer schedules and may shrink on this small heavily-augmented net at 150ep (honest prior: best of the cheap finalists, but could still tie). (2) cosine decays faster initially → marginally less time at high LR; throughput-free so num_epochs identical. (3) the steep end-of-cosine could in principle under-anneal the very last steps, but it finishes at exactly 0 like linear, and EMA denoises the tail.

## Sources
EXP-012 04-analysis.md (flagged schedule-shape); project-insights Medium (anneal-tail dominates, EXP-001); knowledge/references/fast-cifar10-recipes.md (Bag of Tricks cosine); MosaicML LR benchmark (cameronrwolfe substack); SGDR arXiv:1608.03983; train.py:286-290.
