# Proposal idea-02: One-cycle schedule SHAPE — cosine decay & extended low-LR tail

## Core change (train.py only)
The current LR schedule is a **linear triangular one-cycle** (`train.py:286-290`): ramp 0→PEAK over the first `PCT_START`(=0.15) of the time budget, then **linear** decay PEAK→~0. This shape was set in EXP-001 and never tuned. Add a `SCHEDULE` env to vary only the decay shape (warmup unchanged):
- `tri` (default = current linear decay).
- `cos`: cosine decay after warmup — `lr = PEAK * 0.5*(1+cos(pi*(progress-PCT_START)/(1-PCT_START)))`. Spends MORE time at moderate-high LR early then drops steeply — a different exploration/anneal balance.
- `tail`: piecewise-linear with a gentler final tail — decay PEAK→0 but with the last `TAIL_FRAC`(=0.2) of the budget held at a low floor `lr_floor = 0.05*PEAK` (extends the low-LR fine-tuning regime where accuracy concentrates).

```python
p = progress
if p < PCT_START:
    lr = PEAK_LR * p / PCT_START
else:
    q = (p - PCT_START) / (1 - PCT_START)   # 0..1 post-warmup
    if SCHEDULE == "cos":
        lr = PEAK_LR * 0.5 * (1 + math.cos(math.pi * q))
    elif SCHEDULE == "tail":
        lr = PEAK_LR * (1 - q) if q < (1 - TAIL_FRAC) else PEAK_LR * 0.05
    else:  # tri
        lr = PEAK_LR * (1 - q)
```
(import `math`; PCT_START, EMA warmup, TTA gate all unchanged.)

## Mechanism — why this is a DIFFERENT, throughput-free lever
This changes WHERE in training the optimizer spends its steps, NOT the model, the per-step compute, or the regularization. It is **exactly throughput-free** (a scalar LR formula change — num_epochs identical to baseline). The validated pattern "most accuracy lands in the low-LR tail of a completing one-cycle" (EXP-001, project-insights Medium) means the anneal SHAPE plausibly controls the final generalization: a cosine or extended-tail schedule keeps the model longer in the high-curvature exploration phase or the low-LR fine-tuning phase respectively, selecting a different (possibly flatter/better) minimum than the linear ramp.

## Why it targets the limiter
The limiter is the generalization ceiling (project-insights High, EXP-014). EXP-012's analysis EXPLICITLY named "schedule-shape (throughput-free)" as one of "the only untried levers with ceiling clearly above noise" — a standing internal signal that this knob was flagged-but-never-pulled. Unlike capacity/optimizer/aug (all saturated), the decay shape directly governs the anneal-tail generalization that EXP-001 showed dominates final accuracy, and it costs zero throughput (no under-anneal risk — the failure mode behind 5+ prior nulls).

## Throughput
Exactly neutral: a per-step scalar arithmetic change. num_epochs must stay ~150 (verify; any drop indicates a bug, not the schedule). No fused-kernel/op concerns.

## Design — SAME-SESSION multi-cell
- c0: `SCHEDULE=tri` (current) — full-speed same-session anchor (also a regression check: bit-equivalent to baseline).
- cA: `SCHEDULE=cos` — cosine decay, PRIMARY.
- cB: `SCHEDULE=tail` (TAIL_FRAC=0.2, floor 0.05·PEAK) — extended low-LR tail.

## Correctness / EMA / eval
- Only the LR scalar changes; EMA warmup gate (`progress>=0.15`) and flip-TTA gate (`progress>=0.8`) are untouched (they key on `progress`, not LR). Model/eval path byte-identical.
- `SCHEDULE=tri` must reproduce the current baseline behavior exactly (regression smoke: LR trace identical to the existing formula).
- No new params, no VRAM change, no dtype concerns.
- Smoke: LR trace for each schedule — monotone decay after warmup, hits the right floor, `tri` matches baseline at sampled progress points.

## Verification
- Best schedule cell ≥ **96.48** AND > same-session c0 by >0.1pp, replicated with a mandatory confirmation re-run on any apparent win (low-c0-draw lesson).
- num_epochs ≈ 150 (must stay full — throughput-free premise); ep25 sane (cos/ tail change the early-LR so ep25 WILL differ from c0 — judge full-anneal best≈final, not ep25 parity); fully annealed.
- Integrity: train.py-only; prepare.py byte-unchanged; ≤1 eval/epoch; seed 42; summary best == per-epoch max; `tri` ≡ baseline smoke.
- ON A WIN: bake the winning schedule as default.

## Hypothesis
A cosine or extended-low-LR-tail anneal selects a better-generalizing minimum than the linear triangular decay and lifts best_test_acc ≥96.48 over the same-session control, throughput-free at ~150 epochs. If all schedules tie, the linear one-cycle is already near-optimal for this net/budget and the anneal-shape lever is exhausted — strengthening the "genuine ceiling" diagnosis.

## Effort: low. Risk: (1) a well-tuned triangular one-cycle is already strong; cosine/tail rarely beat it by >0.1pp on CIFAR fast-training (honest prior — this may be the most-likely-to-tie of the finalists, but it is genuinely untried and EXP-012-flagged); (2) cos starts the decay slower → could leave slightly less anneal time, but throughput-free so num_epochs identical; (3) the tail floor (0.05·PEAK) could under-anneal the very end — mitigated by keeping the floor low and EMA denoising.
## Sources: EXP-012 04-analysis.md (explicitly flagged schedule-shape as an untried lever with ceiling above noise); project-insights Medium (most gain in low-LR tail, EXP-001); train.py:282-290; standard one-cycle/cosine LR literature (Smith 2018, Loshchilov SGDR arXiv:1608.03983).
