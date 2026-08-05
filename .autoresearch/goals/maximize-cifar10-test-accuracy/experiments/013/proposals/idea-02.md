# EXP-013 idea-02: One-cycle RESHAPE — cosine post-peak decay + earlier peak (PCT_START 0.15→0.10)

## 1. Summary

Replace the **linear** post-peak LR decay in `train.py`'s training loop with a **cosine**
decay to ~0, and (in a separate cell) also move the peak earlier by shortening the ramp
(`PCT_START` 0.15→0.10) with `EMA_WARMUP_FRAC` re-aligned to the new `PCT_START`. The ramp
shape (linear 0→peak) is unchanged; only the decay limb and, optionally, the ramp fraction
change. Throughput is byte-for-byte unaffected — this only changes the scalar `lr` computed
each step, so it **cannot under-anneal** (the failure mode that killed EXP-005/EXP-007).

The exact change is to the `else` branch of the LR block (train.py lines 286–292). Current:

```python
progress = min(1.0, total_training_time / TIME_BUDGET_S)
if progress < PCT_START:
    lr = PEAK_LR * progress / PCT_START
else:
    lr = PEAK_LR * (1.0 - progress) / (1.0 - PCT_START)   # LINEAR decay
```

Proposed decay limb (cosine half-wave from peak at `progress=PCT_START` to 0 at `progress=1`):

```python
import math  # stdlib, top of file
...
if progress < PCT_START:
    lr = PEAK_LR * progress / PCT_START
else:
    decay = (progress - PCT_START) / (1.0 - PCT_START)    # 0→1 over the tail
    lr = PEAK_LR * 0.5 * (1.0 + math.cos(math.pi * decay)) # COSINE decay, →0
```

I propose a **3-cell same-session read** to clear the noise floor (see §5): a same-session
baseline (linear, PCT_START 0.15), cosine-only (PCT_START 0.15), and cosine + PCT_START 0.10
(EMA_WARMUP_FRAC→0.10). The cosine+0.10 cell is the configuration most likely to clear +0.10pp;
cosine-only isolates the shape effect.

## 2. What it targets

The named limiter (EXP-001 Pattern, Medium importance): **"Most accuracy gain arrives in the
low-LR tail of a completing one-cycle"** — `experiments/001/04-analysis.md` shows 89.9% at 55%
progress climbing to 95.2% by the end as LR→0. The current schedule already fully anneals
(EXP-010/EXP-011 confirm the tail completes, peaks then dips), so the lever is not "anneal more"
but **how the LR traverses the low-LR tail where the accuracy concentrates**.

Two coupled targets:
- **Low-LR settling.** A cosine decay reaches small LR values *later in absolute time* than
  linear near the very end but, more importantly, has a vanishing *slope* as LR→0
  (d(lr)/d(progress) → 0 at progress=1), so the final ~10% of training spends more steps at
  genuinely small LR — the regime EXP-001 identifies as where accuracy lands.
- **EMA tail variance.** The weight EMA (decay 0.998, EXP-002, +0.50pp) averages the last
  ~1/(1−0.998) ≈ 500 iterates. A flatter, lower-variance LR tail gives the EMA a tighter cloud
  of iterates to average — directly the mechanism EXP-002 credits ("a smoother tail helps the
  EMA"; `experiments/002/04-analysis.md`).

## 3. Reasoning — confronting the EXP-011 reviewer critique head-on

The EXP-011 idea-03 reviewer (`experiments/011/proposals/idea-03.md` §3) made a precise and
correct objection that I must not paper over:

> "Cosine sits *above* linear in the early decay (stays near peak longer) and *below* it late.
> If the net actually wanted more low-LR time earlier, cosine could marginally hurt."

This is geometrically true. For the half-cosine `0.5(1+cos(πd))` vs linear `(1−d)` over the
tail fraction `d∈[0,1]`:
- at d=0.5 (tail midpoint): cosine = 0.5, linear = 0.5 — they **cross exactly at the midpoint**;
- for d<0.5 (early tail): cosine > linear (holds LR higher, more exploration);
- for d>0.5 (late tail): cosine < linear (drops LR faster toward 0, more low-LR settling time).

So the honest characterization is: **cosine does NOT simply add low-LR time — it trades early-tail
exploration for late-tail settling, pivoting at the tail midpoint.** Whether this helps depends
on which half of the tail is binding. The EXP-001 evidence (gain concentrated as LR→0, i.e. the
*late* tail) argues the late-tail settling is what is binding, which favors cosine — but only
weakly, because the schedule already reaches ~0 under linear too. This is exactly why the
HONEST PRIOR (cosine-vs-linear ≤0.1–0.2pp on an already-annealing one-cycle) holds, and why
cosine-only is expected to sit near the noise floor.

**The shape critique is precisely why cosine-only is the wrong sole test, and why I pair it with
PCT_START 0.10.** Moving the peak earlier (15%→10% of budget) lengthens the *entire* decay limb
from 85%→90% of the budget — a ~6% longer tail in absolute time — and this lengthening is
**shape-agnostic** (it adds low-LR time whether the limb is linear or cosine). Combined with
cosine's late-tail steepening, cosine+0.10 is the configuration that most directly buys "more
late-tail settling time," which is the mechanism the limiter points to. The two levers are
complementary, not redundant: PCT_START stretches the tail; cosine reshapes how it is traversed.

Counter-risk on PCT_START 0.10: shortening the ramp slightly under-warms the high-LR exploration
phase (peak reached after 30s instead of 45s of training time). On a net that converges in ~150
epochs this is a small perturbation, but it is genuinely two-sided (EXP-011 idea-03 rated the
shorter-ramp cell −0.05 to +0.15pp). The same-session cosine-only cell is the control that
attributes any movement to ramp-length vs shape.

**EMA alignment (load-bearing).** `EMA_WARMUP_FRAC=0.15` currently equals `PCT_START` by design
(train.py line 29 comment: "start EMA once LR ramp completes (matches PCT_START)"). If I change
PCT_START to 0.10 I **must** set `EMA_WARMUP_FRAC=0.10` in the same cell, else the EMA would
start 5% of the budget *after* the peak — silently changing the EMA horizon as a second variable
and confounding the read. The cosine-only cell keeps both at 0.15. This is the single most
important not-to-forget detail; the EXP-011 reviewer flagged the identical confound (§4).

**Why this is worth a slot despite the small prior:** schedule SHAPE was set in EXP-001 and
never revisited across EXP-002→012 (whitening, EMA, capacity, aug, optimizer, WD — all left the
decay limb linear). It is the last throughput-free training-side lever with a ceiling that is
*plausibly* above noise and that has *never* been measured here. The downside is bounded (a
scalar formula; cannot crash, cannot under-anneal). The realistic expectation is the upper end
of the cosine-only prior plus the tail-lengthening of PCT_START 0.10: best estimate ~+0.05 to
+0.15pp, i.e. a coin-flip against the +0.10pp bar, which is why the multi-cell same-session
design (not a single run) is essential to read it at all.

## 4. Sources

- **EXP-001 tail pattern** (`experiments/001/04-analysis.md`; learnings §Patterns Medium):
  "Most accuracy gain arrives in the low-LR tail of a completing one-cycle" — 89.9% @55%
  progress → 95.2% by end. This is the limiter the reshape targets.
- **EXP-002 EMA** (`experiments/002/04-analysis.md`; learnings §Patterns High): weight
  EMA(0.998, use_buffers) +0.50pp; tail smoothness is the mechanism the EMA exploits. EMA
  warmup is keyed to PCT_START (train.py L29).
- **EXP-011 idea-03** (`experiments/011/proposals/idea-03.md` §3–4, §Expected): the prior
  cosine-reshape proposal (not selected); supplies the exact shape critique and the
  EMA-alignment confound this proposal answers, plus the calibrated effect-size ranges.
- **fast-cifar10-recipes** (`knowledge/references/fast-cifar10-recipes.md` L12, L27): one-cycle
  triangular lineage; "Bag of Tricks" (He et al. CVPR 2019, arXiv:1812.01187) lists cosine
  annealing as a standard ingredient; SGDR (Loshchilov & Hutter) is the cosine-decay origin.
- **Noise floor** (`experiments/006/04-analysis.md`; learnings §Protocol High): ~0.1pp
  run-to-run jitter from time-budgeted epoch-count variation at fixed seed — mandates a
  same-session baseline cell, not the stored 96.38.

## 5. Estimated Effort: LOW

~5-line edit to the LR `else` branch plus one `import math` at the top. No new deps (`math` is
stdlib). No architecture, optimizer, data, or eval changes. Suggested **same-session 3-cell
read** (each ~7.5 min wall, well under the 10-min kill; run sequentially on GPU 1):

| Cell | PCT_START | EMA_WARMUP_FRAC | Decay | Role |
|------|-----------|-----------------|-------|------|
| C0 (baseline) | 0.15 | 0.15 | linear | same-session reference vs noise floor |
| C1 (cosine)   | 0.15 | 0.15 | cosine | isolates the SHAPE effect |
| C2 (cosine+early) | 0.10 | 0.10 | cosine | the configuration most likely to clear +0.10pp |

Decision rule: require C2 (or C1) to beat **C0 same-session** by ≥0.10pp to count — do not
compare to the stored 96.38 (different host-load draw). Read `num_epochs` on every cell to
confirm throughput is unchanged (all three should fit ~150 epochs; any divergence means
host-load contamination, not a real effect). Watch ep25 test_acc (~92.3 EXP-008 ref) on C2 for
ramp-under-warming.

## 6. Risk Assessment

- **Most likely outcome: within noise.** The honest prior is cosine-vs-linear ≤0.1–0.2pp on an
  already-fully-annealing one-cycle, and the schedule here demonstrably completes (EXP-010/011).
  Cosine-only (C1) is expected ~+0.05pp — below the bar on its own. The whole bet rides on C2's
  tail-lengthening pushing the combined effect over +0.10pp, which is roughly a coin flip. This
  is a low-cost probe of the last untouched training-side axis, not a high-confidence win.
- **Shape pivot could hurt (the EXP-011 critique).** If the binding regime is *early*-tail
  exploration rather than late-tail settling, cosine-only could be flat-to-slightly-negative
  (it holds LR higher than linear in the first half of the tail). C1 vs C0 measures this
  directly; if C1 < C0, the shape hypothesis is falsified and only C2's tail-lengthening (not
  cosine itself) can carry a win.
- **EMA_WARMUP_FRAC misalignment (the avoidable failure).** Forgetting to set
  `EMA_WARMUP_FRAC=0.10` in C2 turns it into a silent two-variable change (ramp length AND EMA
  horizon), making any result uninterpretable. This is the single most important implementation
  detail; it must be set in lockstep with PCT_START. C0/C1 keep both at 0.15.
- **PCT_START 0.10 under-warms exploration.** Reaching peak after 30s of training instead of 45s
  could weaken the high-LR phase; two-sided (EXP-011: −0.05 to +0.15pp). Mitigated by the C1
  control and the ep25 watch.
- **No crash/under-anneal risk.** A scalar LR formula cannot reduce throughput, cannot truncate
  the tail, and `0.5(1+cos(πd))` is bounded in [0, PEAK_LR] and monotone-decreasing on the tail
  — it reaches exactly 0 at progress=1, same endpoint as linear.

## 7. Concrete train.py code sketch

**Add to the stdlib imports (top of file, ~line 2):**

```python
import gc
import math   # NEW — stdlib, no new dependency
import time
```

**Replace the LR block (train.py lines 286–292). The cosine limb shares the linear limb's exact
endpoints — peak at `progress=PCT_START`, 0 at `progress=1` — so the schedule stays continuous
and fully-annealing:**

```python
progress = min(1.0, total_training_time / TIME_BUDGET_S)
if progress < PCT_START:
    lr = PEAK_LR * progress / PCT_START                       # linear ramp (unchanged)
else:
    decay = (progress - PCT_START) / (1.0 - PCT_START)        # 0 at peak → 1 at budget end
    lr = PEAK_LR * 0.5 * (1.0 + math.cos(math.pi * decay))    # COSINE decay, peak → 0
for g in optimizer.param_groups:
    g["lr"] = lr
```

**For the cosine+early cell (C2 only), change two hyperparameters together (train.py lines 25,
29) — keep them equal so EMA still starts exactly at peak:**

```python
PCT_START = 0.10          # was 0.15 — earlier peak, ~6%-longer absolute tail
EMA_WARMUP_FRAC = 0.10    # was 0.15 — MUST track PCT_START (train.py L29 invariant)
```

Sanity checks at the formula's endpoints: at `progress=PCT_START`, `decay=0`,
`cos(0)=1` → `lr=PEAK_LR` (continuous with the ramp top). At `progress=1`, `decay=1`,
`cos(π)=−1` → `lr=0` (same terminal as linear). At the tail midpoint `decay=0.5`,
`cos(π/2)=0` → `lr=0.5·PEAK_LR`, equal to linear there — confirming the documented mid-tail
crossover.
