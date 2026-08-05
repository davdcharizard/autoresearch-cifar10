# Idea-03: Reshape the one-cycle LR anneal to exploit the low-LR tail

## Summary

The LR schedule is a time-based triangular one-cycle: linear ramp 0→`PEAK_LR` over the
first `PCT_START`=15% of the 300s training budget, then **linear** decay `PEAK_LR`→~0 over
the remaining 85%. Its shape (peak 0.4, pct_start 0.15, linear) was fixed in EXP-001 and has
never been revisited, despite the recipe since acquiring whitening (EXP-003), ReZero capacity
(EXP-004), and much stronger augmentation (EXP-008, +0.38pp). A validated project pattern is
"most accuracy gain arrives in the low-LR tail of a completing one-cycle." Reshaping the
anneal is **throughput-free** (pure schedule arithmetic in the hot loop — zero added FLOPs,
zero epoch cost), so it directly targets where accuracy is made without risking the
under-anneal failure that has sunk every capacity experiment (EXP-005, EXP-007).

The honest framing: this is a **low-cost, modest-upside** lever. Cosine-vs-linear on an
*already-fully-annealing* one-cycle is typically ≤0.1–0.2pp, which sits right at this goal's
~0.1pp noise floor. The recommendation below is therefore a **small, tightly-scoped sweep**
(not a single guess), with cosine decay as the primary, highest-confidence single change and
a 3-point peak-LR probe as a cheap rider, because a clean win here needs to clear noise and a
sweep maximizes the chance one cell does.

## Mechanism

Named limiter (from diagnosis): the net is **regularization-bound with a ~4× epoch surplus**,
and **accuracy concentrates in the low-LR tail** (EXP-001: 89.9% @55% progress → 95.2% by end
as LR→0). The schedule shape governs *how much budget is spent in that productive low-LR
regime* and *how smoothly the model approaches the minimum*. Two distinct mechanisms:

1. **Cosine decay (primary).** Linear decay spends LR-time uniformly. A cosine from peak
   spends **more wall-time at moderate-to-low LR** and approaches 0 with a vanishing slope
   (`dlr/dprogress → 0` as progress→1). This is the canonical super-convergence finish (Smith
   one-cycle, Loshchilov-Hutter SGDR cosine): the gentler final approach lets the iterate
   settle into a flatter/wider basin, and — crucially here — it **synergizes with the weight
   EMA**. EMA (decay 0.998, EXP-002, +0.50pp) averages the tail iterates; a smoother,
   slower-moving low-LR tail gives EMA a lower-variance sequence to average, which is exactly
   the "denoised low-LR-tail average" the EMA was added to capture. Linear decay's constant
   slope keeps the iterate moving at a fixed rate right up to near-0, giving EMA a slightly
   noisier tail.

2. **Peak-LR / ramp reshape (rider probe).** The peak (0.4) and ramp length (0.15) were set
   in EXP-001 *before* the frozen whitening front-end (EXP-003) and stronger aug (EXP-008).
   Whitening pre-conditions the input covariance (decorrelated, unit-variance directions),
   which generally **raises the stable LR ceiling**, and stronger aug resists the overfitting
   a hotter peak would otherwise cause. A modestly higher peak (0.5) with the longer cosine
   tail could extract more from the early high-LR exploration phase, then anneal it away. This
   is the lower-confidence half (peak changes carry a stability risk, see Risks).

Causal chain to the metric: smoother/longer low-LR tail → lower-variance, flatter-basin final
iterate → EMA averages a cleaner tail → higher `best_test_acc`. No epoch cost, so unlike
capacity changes there is no under-anneal counter-force.

## Concrete code plan

All changes are in `train.py`, in the LR block of the training loop (lines 286–290) and the
hyperparameter constants (lines 21–30). The schedule stays **time-based on `progress`** so it
fully anneals regardless of host throughput (the EXP-001 invariant; protects against the
shared-host throughput confound that hit EXP-007/EXP-010).

### Primary change: cosine post-peak decay

Add `import math` (top of file, near line 1–2). Replace the decay branch (lines 286–290):

```python
import math  # add at module top
```

```python
# Time-based one-cycle LR: linear ramp 0 -> PEAK over the first PCT_START of the
# budget, then COSINE decay PEAK -> ~0 over the remaining (1 - PCT_START). Keyed on
# elapsed *training* time so the anneal completes regardless of throughput.
progress = min(1.0, total_training_time / TIME_BUDGET_S)
if progress < PCT_START:
    lr = PEAK_LR * progress / PCT_START
else:
    decay_frac = (progress - PCT_START) / (1.0 - PCT_START)   # 0 -> 1 over the tail
    lr = 0.5 * PEAK_LR * (1.0 + math.cos(math.pi * decay_frac))
```

At `decay_frac=0` (peak just reached) this gives `lr=PEAK_LR`; at `decay_frac=1`
(budget end) `lr=0` exactly — same endpoints as the linear schedule, only the shape between
changes. This is the cleanest single-variable test (peak, ramp, EMA gate, TTA gate all
unchanged), so attribution is unambiguous.

### Rider probe (sweep cells, not committed unless they win)

The sweep harness (see `experiments/010/02-sweep.md` precedent) made `PEAK_LR_MUON`
env-overridable so trials set values at runtime without editing the file. Mirror that: make
`PEAK_LR` and `PCT_START` env-overridable so the sweep can vary them on top of the cosine
shape without further edits:

```python
import os
PEAK_LR = float(os.environ.get("PEAK_LR", 0.4))
PCT_START = float(os.environ.get("PCT_START", 0.15))
```

Sweep grid (all on the cosine shape, full-budget trials, real `best_test_acc`):

| Cell | PEAK_LR | PCT_START | Decay | Purpose |
|------|---------|-----------|-------|---------|
| C0 (baseline) | 0.4 | 0.15 | linear | reference (current `train.py`) |
| C1 (primary) | 0.4 | 0.15 | **cosine** | shape-only, highest-confidence |
| C2 | 0.4 | **0.10** | cosine | shorter ramp → longer tail |
| C3 | **0.5** | 0.15 | cosine | hotter peak (whitening-tolerance probe) |

Four full-budget trials × ~445s wall each ≈ 30 min — within one experiment loop. Winner cell
gets committed to `train.py` as static defaults (env overrides removed for the committed
version, matching EXP-010's pattern of baking the winner in).

## Recommended config + defaults

**If forced to a single change (highest confidence, cleanest attribution): C1 — cosine decay
only**, `PEAK_LR=0.4`, `PCT_START=0.15` unchanged. This isolates shape from magnitude and is
the change most directly motivated by the EMA-synergy + super-convergence literature.

**Recommended as run: the 4-cell sweep above (C0–C3).** Rationale: a shape-only change is
expected at ≤0.1–0.2pp (near noise), so a single trial risks an ambiguous "within-noise"
result. The sweep is cheap (throughput-free cells, ~30 min total) and lets the primary (C1)
plus two cheap riders (C2 longer tail, C3 hotter peak) compete; the riders are the realistic
path to clearing the +0.10pp bar if pure cosine alone does not. C0 is re-run as a same-session
reference to neutralize the throughput/epoch-count jitter that the noise-floor finding warns
about (compare cells at matched host load, not against the stored 96.38).

## Interaction with EMA warmup and TTA gate

- **EMA warmup** (`EMA_WARMUP_FRAC`=0.15, line 29) is tied to `PCT_START` "to start EMA once
  the LR ramp completes." For C2 (`PCT_START`=0.10) the comment's invariant breaks: EMA would
  start at 15% but the ramp now ends at 10%, leaving a 5% high-LR window where EMA is *not*
  averaging. To keep attribution clean, **C2 must also set `EMA_WARMUP_FRAC=PCT_START`** so
  EMA still begins exactly when the ramp completes. For C1 and C3 (`PCT_START` unchanged) the
  EMA gate is untouched — no coupling change. This is the one place reshaping is NOT purely
  local; pre-register it so the analysis attributes C2's result to the *combined* ramp+EMA
  shift, not ramp alone.
- **TTA gate** (`TTA_START_FRAC`=0.8, line 30) is keyed on `progress`, independent of decay
  shape — the final-20% window is the same wall-time regardless of cosine vs linear. No change
  needed. The cosine tail makes LR *lower* than linear inside that window (cosine is below the
  linear line for most of the back half), which is consistent with TTA wanting a settled
  model — no adverse interaction expected.
- **C3 hotter peak (0.5)** does not change *where* EMA starts (still 15%) but raises the LR
  *during* the no-EMA ramp; fine, EMA only averages post-ramp iterates. No gate change.

## Risks & de-risking

1. **Within-noise (most likely failure).** Shape-only cosine on an already-annealed one-cycle
   is often ≤0.1pp; the +0.10pp bar sits at the ~0.1pp noise floor (EXP-006 finding). De-risk:
   run C0 same-session as reference so the comparison is at matched host throughput; require
   the winning cell to beat *same-session C0* by ≥0.10pp AND show a tail-shape signature
   (see read), not just a single-number bump. Honest stance: this is the dominant risk and the
   reason expected value is modest.
2. **Hotter peak (C3) destabilizes.** Higher LR is the documented failure mode on this loop
   (EXP-009 Muon diverged at peak 0.24). Mitigant: SGD-Nesterov is far more robust than
   weight-renorm Muon, and 0.4→0.5 is a small step on an optimizer already stable at 0.4; the
   whitening front-end + LS 0.2 + strong aug all raise tolerance. De-risk: read **ep25
   stability** — if C3's ep25 test_acc is materially below C0's (~92.3, EXP-008 ref) or the
   smoothed train loss spikes/NaNs, mark C3 unstable and drop it (do not commit). C3 is the
   most likely cell to fail; C1 carries essentially no stability risk.
3. **Cosine *under*-uses the early budget.** Cosine sits *above* linear in the early decay
   (stays near peak longer) and *below* it late. If the net actually wanted more low-LR time
   earlier, cosine could marginally hurt. This is why C2 (shorter ramp → earlier into the
   decay) is included as a hedge.
4. **EMA-coupling confound on C2** (covered above): pin `EMA_WARMUP_FRAC=PCT_START` so C2 is
   ramp+EMA-aligned, not a silent two-variable change.

## Expected effect (pp + reasoning)

- **C1 (cosine only):** +0.0 to +0.15pp. Best estimate ~+0.05–0.10pp. Reasoning: cosine-vs-
  linear gains on a completing one-cycle are well-documented but small once the schedule
  already fully anneals (which this one does — EXP-001/EXP-010 confirm the tail completes).
  The EMA-synergy argument is the reason to expect the upper half of that range rather than 0.
- **C2 (shorter ramp):** −0.05 to +0.15pp. Lengthening the tail could help (more low-LR time)
  or the shorter ramp could under-warm the high-LR phase; genuinely two-sided.
- **C3 (hotter peak):** −0.3 to +0.2pp. Highest variance: could extract real early-exploration
  gain *or* destabilize. Expected ~0 with fat tails.

**Realistic overall:** the sweep's *best cell* lands ~+0.05–0.15pp over same-session C0, i.e.
a coin-flip on clearing the +0.10pp bar. This is an honestly modest lever — its appeal is the
near-zero cost and clean attribution, not a large expected delta. It is worth running because
(a) the schedule shape is genuinely stale (set 4 recipe-generations ago), (b) it cannot
under-anneal (the failure mode that killed EXP-005/007), and (c) it is among the few remaining
throughput-free levers after the optimizer axis was declared exhausted (EXP-010).

## Pre-registered success / failure read

Measure each cell at full host throughput (confirm GPU 1 free; EXP-010 was throughput-
confounded mid-sweep). Record per cell: `best_test_acc`, `num_epochs`, ep25 test_acc,
the eval trajectory at ep25/50/75/100/tail, and the final-tail behavior (peaked-then-dipped
vs still-climbing).

- **SUCCESS (commit winning cell):** best cell's `best_test_acc` ≥ same-session C0 + 0.10pp
  AND ≥ 96.48 absolute, with a coherent mechanism signature — for C1, a *higher tail floor*
  (later epochs sit above C0's at matched epochs) consistent with a cleaner EMA average, not a
  single lucky epoch. `num_epochs` must be within noise of C0 (schedule reshape is
  throughput-free; a large epoch drop would signal an unrelated confound).
- **FAILURE / within-noise (discard):** best cell < C0 + 0.10pp or < 96.48 → schedule shape is
  near-exhausted at the current recipe; the linear one-cycle is already close to optimal and
  this lever does not clear noise. Record as no-improvement; do not re-test shape variants
  standalone (fold any future shape tweak in as a free rider on a training-side win that
  itself clears >0.1pp, per the EXP-006 TTA precedent).
- **C3 instability gate (pre-registered):** if C3 ep25 << C0 ep25 (≳1pp below) or train loss
  diverges, mark C3 unstable, exclude from the winner decision, and confirm SGD's peak ceiling
  on this net is ≤0.5 (a reusable finding).

## Effort

**Low.** The code change is a ~5-line edit to one branch of the LR block plus a 2-line env-
override for the sweep. The sweep harness pattern already exists (`experiments/010/02-sweep.md`,
`sweep.py`). Four full-budget trials ≈ 30 min wall total. No new deps (`math` is stdlib), no
architecture change, no throughput risk. Cleanest and cheapest experiment class on this goal.
