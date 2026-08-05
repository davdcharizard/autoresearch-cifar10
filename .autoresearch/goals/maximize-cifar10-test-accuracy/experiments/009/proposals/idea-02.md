# Proposal (idea-02): Retune the EMA horizon to track the still-rising annealed tail — `EMA_DECAY 0.998 → 0.995`

## One-line

The EXP-008 tail is still monotonically rising at the budget end, and the **evaluated weights are an EMA, not the raw iterate** — a 0.998 EMA lags the improving late iterate by ~3.6 epochs, dragging the scored model back toward worse, higher-LR weights. Shortening the EMA horizon (`EMA_DECAY 0.998 → 0.995`, half-life ~3.6 ep → ~1.4 ep) lets the evaluated average track the late gains, capturing under-anneal headroom **at zero throughput cost**. This is a single-variable change to one constant.

## Why this knob, not the LR knobs (a)–(d)

The idea brief lists five candidate throughput-free levers: (a) lower PEAK_LR, (b) lower PCT_START, (c) cosine/lingering decay shape, (d) nonzero LR floor, (e) faster EMA. After reading the actual training loop (`train.py:275-349`) I recommend **(e), `EMA_DECAY 0.998 → 0.995`**, and explicitly reject (a)–(d) for this experiment. The reasoning is mechanistic, not a preference:

**The thing actually evaluated each epoch is the EMA, not the raw model.** `train.py:343-349`: once `ema_started`, `eval_target = ema_model` and `evaluator.evaluate` scores `ema_model`. So `best_test_acc` is a property of the **EMA trajectory**, which is a low-pass filter of the raw iterate. Any LR-schedule reshape (a–d) acts on the *raw* iterate and only reaches the metric *through* the same EMA filter. If that filter is mis-tuned for a rising tail, it caps the realizable gain from ANY tail-anneal improvement. So the EMA horizon is the lever that sits directly on the metric and gates the others.

**Quantifying the EMA lag.** EXP-008 ran 14480 steps / 150 epochs = **96.5 steps/epoch** (`train.py` summary in EXP-008 execute log: 150 epochs / 14480 steps). EMA half-life in steps is `ln(2) / (1 - decay)`:
- decay 0.998 → half-life ≈ 347 steps ≈ **3.6 epochs**
- decay 0.995 → half-life ≈ 139 steps ≈ **1.4 epochs**

EXP-008's observed tail: ep147 96.32 → ep150 96.38, monotone rising, **best == final**. A 3.6-epoch-lagged average evaluated at ep150 is effectively centered near ~ep146–147, i.e. on weights from a **measurably higher-LR, less-annealed** part of the schedule (at progress 0.97, LR ≈ 0.4·0.03/0.85 ≈ 0.014; at progress 0.93 LR ≈ 0.033 — over 2× higher). When the raw iterate is still improving monotonically, a shorter-horizon EMA is centered closer to the (best) final iterate and should score at-or-above the long-horizon EMA. This is the exact regime — a rising, near-converged tail — where the standard "longer EMA = more denoising" intuition reverses, because there is little tail *noise* left to suppress (the +0.06pp/3ep monotone rise shows the late iterate is already smooth) and the dominant error is **horizon bias**, not variance.

**Why NOT (a) lower PEAK_LR:** direction is genuinely uncertain and the risk is asymmetric-bad. One-cycle's high-LR phase is the exploration that sets up the basin the anneal later sharpens; EXP-001 established PEAK_LR=0.4 as the value that delivered the +3.65pp foundation, and EXP-008's *early* trajectory (ep25 92.31, ep50 93.75, ep100 95.13) shows the net is converging healthily, NOT thrashing from too-high LR. Lowering peak to 0.3 trades certain exploration for speculative extra anneal time and could easily be wrong-signed. Higher variance than (e), weaker mechanism.

**Why NOT (b) lower PCT_START:** moving 0.15→0.10 shifts only 5% of the budget (~15s, ~7 epochs) from ramp into decay. But the ramp is NOT wasted — it is the warmup the BN-heavy net needs, and `EMA_WARMUP_FRAC=0.15` is deliberately tied to `PCT_START` (`train.py:29`, comment "matches PCT_START"; gate at `train.py:308`). Decoupling them (lowering PCT_START without EMA_WARMUP_FRAC) would start EMA *during* the ramp, polluting the average with high-LR weights — a confound. Lowering both is then a two-variable change. The net redistribution is tiny and the mechanism for it helping the *tail specifically* is weak.

**Why NOT (c) cosine / lingering decay:** plausible but it is the SAME class of fix as (e) (spend more effective time at low LR) delivered through a *more invasive* code change (rewriting the decay branch `train.py:289-290`), and it STILL must pass through the EMA filter to reach the metric. (e) is the smaller, higher-leverage version of the same hypothesis. If (e) wins, (c) becomes a clean follow-up; if (e) loses, (c) is unlikely to clear noise either.

**Why NOT (d) nonzero LR floor:** the idea brief itself flags this as "usually worse," and the literature is consistent — annealing to ~0 is where CIFAR one-cycle accuracy is set (EXP-001 key learning: "always anneal to ~0"). A floor leaves residual gradient noise in the final weights. Rejected.

## Mechanism → metric (the causal chain)

1. EXP-008 left a **monotone-rising, near-converged tail** (ep147→150: 96.32→96.38, best==final) — the diagnosis's "capacity/regularization under-anneals; epoch surplus" signature.
2. The scored model is the **EMA** (`train.py:345`), a low-pass filter with half-life 3.6 epochs at decay 0.998.
3. On a rising tail with little residual noise, that 3.6-epoch lag is **bias toward worse, higher-LR weights**, not useful denoising — it under-reads the best (final) iterate.
4. Shortening to decay 0.995 (half-life 1.4 ep) re-centers the evaluated average closer to the final iterate, recovering the part of the rising tail the long EMA was discarding.
5. → higher `best_test_acc`, and a **diagnostic prediction**: the best==final gap should persist or the final-epoch EMA acc should rise relative to mid-tail, since we are reading a less-lagged point on the same rising curve.

## Concrete change in THIS codebase

Single constant edit in `train.py:28`:

```python
EMA_DECAY = 0.998  # short-horizon weight EMA (denoised low-LR-tail average)
```
→
```python
EMA_DECAY = 0.995  # shorter horizon (~1.4 ep half-life) to track the rising annealed tail
```

That constant flows unchanged into the EMA construction at `train.py:255-257`:
```python
ema_model = AveragedModel(
    model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True
).to(...)
```
`get_ema_multi_avg_fn(0.995)` produces the per-step update `new = 0.995*avg + 0.005*param`. **Nothing else changes.** EMA still warms up at `progress >= EMA_WARMUP_FRAC=0.15` (`train.py:308`), still updates every step, still `use_buffers=True` (BN stats EMA-averaged on-budget, no `update_bn` pass), still evaluated in place of the raw iterate (`train.py:343-349`). The LR schedule (`train.py:282-290`), PEAK_LR, PCT_START, optimizer, augmentation, TTA gate, and `torch.manual_seed(42)` are **byte-identical to EXP-008**. This is the cleanest possible single-variable test.

A planner can turn this into a one-line diff. No new constant, no new code path, no new dependency.

## Evidence

- **Code fact (load-bearing):** the metric is computed on `ema_model`, not `model` (`train.py:345`, `eval_target = ema_model`). Confirmed by reading the eval block. This is what makes the EMA horizon a first-class metric lever rather than a minor denoiser.
- **EXP-008 analysis** (`experiments/008/04-analysis.md` §Results obs. 3): "Mild still-rising tail at the end (ep147 96.32 → ep150 96.38, best==final)... the anneal may not be fully complete for the harder-augmented net" — the exact under-anneal this idea targets, and its own pre-registered "Unexplored Avenue" #5 names a tail-anneal tweak.
- **EXP-002 analysis** (`experiments/002/04-analysis.md` §Unexplored, and Patterns High-Importance EMA bullet in `03-experiment-learnings.md`): "EMA decay 0.998 was a first principled guess... a progress-scheduled decay could capture more of the anneal." The 0.998 value was never tuned against the current (harder-augmented, EXP-008) tail shape — it predates EXP-008 by six experiments. That is the mismatch this experiment exploits, analogous to the "PEAK_LR tuned for the lighter-aug recipe" framing but applied to the lever that sits directly on the metric.
- **Step/epoch arithmetic** (EXP-008 execute log: 150 epochs / 14480 steps) gives the concrete 3.6-ep vs 1.4-ep half-life contrast that quantifies the lag — not hand-waving.
- **EXP-001 key learning** (`03-experiment-learnings.md`): "Most accuracy gain arrives in the low-LR tail... always anneal to ~0" — establishes that the tail is where the metric is decided, so a tail-reading filter (the EMA horizon) is high-leverage and the LR floor option (d) is contraindicated.

## Expected magnitude vs the noise floor

Honestly: **small and at risk of being sub-noise.** The realizable headroom is bounded by how much of the rising tail the long EMA currently discards. The tail rises only +0.06pp over the last 3 epochs and is monotone (near-converged), so the EMA's 3.6-epoch lag is reading weights worth at most ~0.1–0.2pp below final. A best-case recovery is therefore ~+0.05 to +0.15pp. The bar is ≥96.48 (+0.10 over 96.38) AND clearly above the ~0.1pp noise floor. So this change is **plausibly real but plausibly sub-noise** — it is on the optimistic edge of clearing the bar, not comfortably over it. I will not overstate it: of the five candidates this has the clearest mechanism and the smallest blast radius, but the headroom itself is intrinsically small because the tail is near-converged.

## Strongest risk

**The assumption that most needs to hold: the tail is rising-but-low-noise, so shorter-horizon EMA reduces bias more than it adds variance.** If, instead, the per-step late iterate is *noisier* than the smooth epoch-level eval prints suggest (epoch evals subsample the trajectory — a monotone epoch sequence can hide step-level jitter), then 0.995 will *under-denoise* and could LOSE accuracy. Faster EMA is strictly a bias/variance trade; I am betting the rising-tail evidence means we are bias-dominated, but a single eval/epoch cannot directly observe the step-level noise, so this is a genuine unfalsified assumption. Secondary risk: the effect is real but ~+0.05pp, under the noise floor, yielding an honest `no-improvement` (the EXP-006 failure mode — a real-but-sub-noise eval-side tweak). Because the seed is fixed (no re-rolling) and host-throughput jitter is ±~0.1pp, a sub-noise true effect is unprovable in one run.

## Verification / falsification

- **Throughput unchanged (must hold for clean attribution):** `num_epochs` must land in the EXP-004/008 band (~142–150) and `total_seconds ≈ 440–450s`. The EMA update is per-step regardless of decay value, so step cost is identical — any epoch-count shift would be host jitter, not the change. If `num_epochs` drops below ~130, treat as a throughput-confounded run, not a verdict on the knob.
- **Mechanism confirmation:** compare the late-epoch EMA trajectory to EXP-008's. The prediction is the tail sits **at or above** EXP-008's (96.32→96.38) and the *final-epoch* EMA acc is at least as high — because a shorter-horizon EMA on a rising curve reads a later (better) point. If the tail comes in *below* EXP-008's, the variance-vs-bias bet was wrong (faster EMA under-denoised) — clean falsification.
- **Falsification of the whole hypothesis:** if `best_test_acc ≤ 96.38` with `num_epochs` in band, the "long EMA was discarding tail gains" mechanism is refuted for this recipe, and the residual under-anneal (if any) must be attacked on the LR side (option c, cosine decay) instead — a well-scoped next experiment.
- **Constraints:** only `train.py` edited; `prepare.py` byte-unchanged; seed `torch.manual_seed(42)` intact; exactly one `evaluator.evaluate` per epoch (unchanged); no new deps. Run on GPU 1 under `timeout 600`.

## Effort

**Low** — one-character-class constant change (`0.998 → 0.995`), no new code path, one training run within the existing harness. Smoke check: confirm `get_ema_multi_avg_fn(0.995)` constructs and one EMA update runs (it is the same API as EXP-002), then the single official run.
