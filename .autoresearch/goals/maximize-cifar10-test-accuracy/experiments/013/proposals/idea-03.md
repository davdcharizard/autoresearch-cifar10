# Proposal idea-03: Peak LR as an implicit regularizer — raise PEAK_LR on the matured recipe

**Goal**: maximize-cifar10-test-accuracy. Baseline **96.38** (EXP-008, commit 07c3760). Bar **≥96.48** (+0.10pp), and must clear the ~0.1pp time-budgeted epoch-jitter noise floor (seed fixed → use a same-session baseline cell, pre-register the read).
**Edit scope**: `train.py` only. `prepare.py` frozen. No new deps. Seeds intact. ≤1 eval/epoch. 10-min wall kill. GPU 1.

---

## 1. Summary

Re-tune the single scalar `PEAK_LR` (line 21 of `train.py`), which was set to **0.4** in EXP-001 at the 95.2% recipe and never re-tuned across the four subsequent recipe maturations (EMA EXP-002, ZCA whitening EXP-003, ReZero EXP-004, stronger aug EXP-008). Run a **same-session 3-cell pre-registered exploration** with `PEAK_LR` env-toggled (exact EXP-012 pattern):

| cell | PEAK_LR | role |
|------|---------|------|
| 0 | **0.4** | same-session baseline (reproduces EXP-008 training exactly) |
| A | **0.5** | +25% peak — primary read |
| B | **0.6** | +50% peak — stress / instability probe |

Everything else byte-identical to EXP-008 (architecture, whitening, EMA, schedule shape PCT_START=0.15, SCALE_OUT=0.125, augmentation, batch 512, WD 5e-4, LS 0.2, seeds). This isolates `PEAK_LR` for clean attribution. The change is **throughput-free** (a constant fed into the existing time-based triangular schedule), so it cannot under-anneal — num_epochs should hold at ~142–150 across all three cells. Win = some cell ≥96.48 AND >same-session cell-0 by ≥0.10pp at num_epochs ≥142.

**Honest framing up front**: this is essentially tuning ONE already-exposed scalar — close to a 1-parameter sweep. The expected effect size is small and may be sub-noise, and there is a genuine instability tail-risk at 0.6 (see §6). It is cheap (low effort) and pre-registered, so it is a reasonable low-cost probe, but it should not be oversold as a likely >0.1pp win.

---

## 2. What it targets (named limiter)

The diagnosis across EXP-009/010/011/012 converges on one conclusion recorded in `03-experiment-learnings.md`: **this whitened ResNet-9 at 300s is regularization-bound, not optimizer- or capacity-bound** (Muon ties SGD → not optimizer-bound; WD-shaping + ReZero-α instrumentation → capacity gate not accuracy-limiting; CutMix redundant with occlusion aug). On a regularization-bound net, the lever that moved the metric most was a regularization lever (EXP-008 stronger aug, +0.38pp).

`PEAK_LR` is a regularization lever via a mechanism distinct from input-aug, weight-decay, and label-smoothing (all of which are now exhausted per EXP-011/012): **SGD gradient-noise-scale implicit regularization**. The covariance of the minibatch-gradient noise injected per step scales with the learning rate; a larger peak LR in the high-LR phase of the one-cycle injects more noise, biasing SGD toward flatter / wider minima that generalize better. So this targets the named regularization limiter through an axis the brainstorm history has not yet touched.

Secondary target: **whitening-enabled LR headroom**. The EXP-003 learning explicitly lists "whitening-enabled higher PEAK_LR" as an untried rider (`03-experiment-learnings.md` line 101: "Untried riders: whitening-enabled higher PEAK_LR, identity-init stem"). The frozen ZCA whitening front-end (`compute_whitening_weight`, `model.whiten`) decorrelates and unit-scales the input patch directions, improving the conditioning of the loss surface seen by the learnable stem `prep` and downstream layers. Better-conditioned curvature → a higher stable peak LR than the original (un-whitened) EXP-001 net that set 0.4. The 0.4 value predates whitening, so the stable ceiling has very plausibly risen.

---

## 3. Reasoning (the causal chain)

**Implicit-regularization mechanism.** In SGD, the per-step weight update is the true gradient plus zero-mean minibatch noise whose covariance scales ∝ (LR² / batch). Classic linear-scaling-rule analyses (Goyal et al. 2017; Smith & Le 2018, "A Bayesian Perspective on Generalization and SGD") frame the effective gradient-noise "temperature" as g ≈ LR·(N/B). Holding batch B=512 and dataset N fixed, raising LR raises the noise temperature in the high-LR plateau of the one-cycle, which is the regime where the implicit-regularization / minimum-selection happens. On a net whose ceiling is set by generalization (not fit), more noise in that phase can push to a flatter minimum and lift test accuracy — the same direction as EXP-008's aug win, via a different knob.

**Linear-scaling / batch-512 argument.** Batch 512 is large relative to the DavidNet lineage's canonical batch (cifar10-fast / DAWNBench used 512 with peak LR in the ~0.4–1.0 "mean-loss lambda" band depending on the exact loss convention and net). The linear-scaling rule says the appropriate LR grows with batch; at batch 512 a peak of 0.4 is on the conservative side of what large-batch theory permits, so there is headroom to test 0.5–0.6 before the large-batch stability limit. (This is a directional plausibility argument, not a precise prescription — the constant in the scaling rule is net- and schedule-dependent, which is exactly why we measure rather than assume.)

**SCALE_OUT interaction (a real caveat, not just a footnote).** `_forward_once` multiplies logits by `SCALE_OUT=0.125` (line 178). Because cross-entropy gradient magnitude scales with the logit scale, SCALE_OUT and PEAK_LR jointly set the *effective* step size on the upstream weights — they partly trade off. This means 0.4 is not a "raw" LR but an LR-at-this-logit-scale; raising PEAK_LR with SCALE_OUT fixed is a clean way to probe whether the current effective step is below optimum. We hold SCALE_OUT fixed to keep the attribution to PEAK_LR clean (changing both at once would confound).

**Whitening-conditioning argument.** See §2 secondary. The 0.4 ceiling was measured on the pre-whitening net (EXP-001); whitening (EXP-003) was added later and improves input conditioning (its motivation is exactly faster/healthier early optimization — EXP-003 saw ep10 jump 81.6→85.5%). A better-conditioned front-end raises the largest stable LR, so the inherited 0.4 is plausibly now sub-optimal-low.

**Why it could still be flat (honest).** Most accuracy on this recipe lands in the low-LR anneal tail (EXP-001 medium-importance learning: "Most accuracy gain arrives in the low-LR tail"). The peak-phase noise temperature is then a second-order knob relative to the anneal, and the one-cycle already anneals to ~0 regardless of peak, so the peak height may simply not bind the final minimum tightly. EXP-012 found the SGD regularization scalars (WD allocation, LS) already tuned — PEAK_LR is the remaining un-swept member of that family, and the prior is correspondingly weak.

---

## 4. Concrete change (files / functions actually read)

All in `train.py`. I read the full file; the touch points are exact:

1. **Imports (top, after line 2 `import time`)**: add `import os` (not currently imported).

2. **Hyperparameter block, line 21**: replace
   ```python
   PEAK_LR = 0.4  # mean-loss one-cycle peak (DavidNet "lambda" convention)
   ```
   with
   ```python
   PEAK_LR = float(os.environ.get("PEAK_LR", "0.4"))  # one-cycle peak (DavidNet "lambda"); env-toggled for the same-session read, default 0.4 == EXP-008 baseline
   ```
   Default `"0.4"` ⇒ an unmodified (no-env) invocation reproduces EXP-008 training behavior exactly; cell-0 confirms this empirically against the stored 96.38.

3. **No other code changes are required for the mechanism.** `PEAK_LR` already flows everywhere it must:
   - Optimizer init `lr=PEAK_LR` (line 246).
   - The time-based triangular schedule in the loop (lines 286–292) computes `lr` from `PEAK_LR` each step and writes it to **every** param group via `for g in optimizer.param_groups: g["lr"] = lr` — so a single scalar drives both ramp and decay. No schedule edit needed.
   - `EMA_WARMUP_FRAC`/`PCT_START`/`TTA_START_FRAC` are independent of PEAK_LR — schedule SHAPE is unchanged, only the peak HEIGHT.

4. **Summary block (after line 373 `best_test_acc` print)**: add self-describing + instability-watch prints
   ```python
   print(f"peak_lr:          {PEAK_LR}")
   print(f"acc_at_ep25:      {acc_at_ep25:.2f}%")   # high-LR-phase instability probe (see §6)
   ```
   To populate `acc_at_ep25`, add near the loop's eval/best block (after line 352) a cheap capture:
   ```python
   if epoch == 25:
       acc_at_ep25 = test_acc
   ```
   and initialize `acc_at_ep25 = 0.0` alongside `best_acc = 0.0` (line 273). ep25 is the EXP-009/EXP-011 convention for the high-LR-phase health check (EXP-009 detected Muon divergence as ep25-100 collapse to ~10–20%; EXP-011 logged ep25 92.31 healthy). If a cell prints a depressed/collapsed `acc_at_ep25`, that is the divergence signature.

A planner can turn this into a diff directly: it is one import, one line replaced with an `os.environ.get`, one `if epoch == 25` capture, two summary prints, one variable init. No architecture, optimizer-structure, RNG, or eval-path change. `num_params` stays 7,784,627. This mirrors EXP-012's env-toggle pattern (`02-plan.md` §Code Changes) exactly, which is verified to run in this env.

---

## 5. Evidence (specifics)

- **EXP-001** (`03-experiment-learnings.md` line 93–95; `04-results.tsv` row 001): set PEAK_LR=0.4 at the 95.22% pre-whitening/pre-EMA/pre-ReZero/pre-strong-aug recipe; the value is the "mean-loss DavidNet lambda" convention and has been carried unchanged through every subsequent maturation. **Stale-scalar premise is factual, not speculative.**
- **EXP-003** (`03-experiment-learnings.md` line 101): explicitly registers "whitening-enabled higher PEAK_LR" as an **untried rider** — this proposal is the pre-registered follow-up to that note. Whitening improved early-optimization health (ep10 81.6→85.5%), the conditioning signal that motivates LR headroom.
- **Regularization-bound diagnosis** (`03-experiment-learnings.md` Failed Approaches, EXP-009/010/011/012): four independent experiments converge that the net is regularization-bound and that input-aug / WD-allocation / LS / optimizer-swap are exhausted. PEAK_LR is the regularization-family scalar EXP-012 did NOT sweep ("the conv/fc wd *level* itself … was never swept" line 62 flags un-swept scalars remain; PEAK_LR is the analogous un-swept LR scalar).
- **Implicit-regularization literature**: Smith & Le 2018 (Bayesian/SGD-noise view, noise scale g∝LR·N/B); Goyal et al. 2017 (linear scaling rule, large-batch LR↑); Li et al. 2019 ("Towards Explaining the Regularization Effect of Initial Large Learning Rates"). These support the directional claim that a higher peak LR in the high-LR phase increases generalization-relevant gradient noise. (General mechanism citations; the magnitude on THIS net is what we measure.)
- **Large-batch headroom**: batch 512 (line 20) with the existing 0.4 peak is on the conservative side of the linear-scaling band for this batch — directional headroom for 0.5–0.6.

---

## 6. Risk assessment (honest)

**Strongest risk — high-LR instability (the assumption that most needs to hold: the net stays stable at 0.5–0.6 under the LONG one-cycle).** EXP-009 (`03-experiment-learnings.md` line 66–68) is the live caution: at peak LR 0.24 the **long 150-epoch triangular schedule's high-LR plateau destabilized BN and the net diverged to ~random through ep25–100, recovering only in the anneal tail** (94.11). Crucial nuance that DE-risks this proposal: EXP-009 was **Muon** with airbench weight-renorm pinning ‖conv‖=√out (a ~24%/step rotation mechanism specific to orthogonalized+renormed updates) — SGD-Nesterov has no such weight-norm pinning and is far more robust to high LR. The EXP-009 divergence is not expected to transfer to plain SGD at 0.5–0.6. BUT EXP-009 also flags the *general* lesson that **this net's long schedule is less LR-tolerant than airbench's short 8-epoch no-plateau schedule** — the long high-LR plateau is the vulnerable regime. So the instability tail-risk is real, concentrated at cell-B (0.6). **Mitigation**: the `acc_at_ep25` print is the pre-registered watch — a depressed ep25 (well below EXP-011's healthy 92.31, e.g. <85%) signals high-LR instability for that cell; treat such a cell as a stability failure (abort/exclude), not a clean read. Cell-A (0.5) is the lower-risk primary; cell-B (0.6) is the stress probe.

**Sub-noise risk (the most likely benign failure).** This is near a 1-parameter sweep of an already-tuned regularization family. The peak-phase noise is second-order to the anneal tail where accuracy lands (EXP-001), and EXP-012 found the SGD regularization scalars already tuned. The most probable outcome is **flat within the ~0.1pp noise floor** — cells A/B within ±0.1pp of cell-0. The same-session 3-cell design + pre-registered ≥0.10pp-over-cell-0 read is exactly what prevents a sub-noise jitter from being mis-read as a win. I state plainly: I do not have strong prior confidence this clears the bar; its value is that it is cheap, pre-registered, and closes an explicitly-flagged untried rider.

**Throughput confound (low).** Throughput-free, so num_epochs should hold ~142–150 across cells; if shared-host load drifts >5 epochs across the sequential cells, re-run the affected cell adjacent to cell-0 (EXP-012 sequential-drift guard). Abort any cell <110 epochs (under-anneal / contention).

**Over-fit-to-noise on a "win".** If cell-A lands in [96.48, 96.55), require a confirmation re-run of {winning cell, cell-0} back-to-back (the ≥0.10pp gap must survive; EXP-012 thin-winner protocol).

---

## 7. Estimated effort: **LOW**

One import, one `os.environ.get` line, one ep25 capture, two prints. No architecture/optimizer-structure/RNG/eval change. Three same-session cells back-to-back (~445–460s wall each → ~25 min) + at most one confirmation pair. This is the cheapest class of experiment (a scalar probe with a same-session read), comparable to EXP-012's mechanics minus the optimizer-group split.

---

## 8. train.py sketch (env-toggled PEAK_LR, EXP-012 pattern)

```python
import os          # NEW (top of file, with the other imports)
# ...
# hyperparameter block, replacing line 21:
PEAK_LR = float(os.environ.get("PEAK_LR", "0.4"))  # env-toggled peak; default 0.4 == EXP-008 baseline

# loop init (with best_acc = 0.0, ~line 273):
best_acc = 0.0
acc_at_ep25 = 0.0

# after the best-acc update (~after line 352):
if epoch == 25:
    acc_at_ep25 = test_acc   # high-LR-phase health probe (EXP-009 divergence signature)

# summary block (after the best_test_acc print, ~line 373):
print(f"peak_lr:          {PEAK_LR}")
print(f"acc_at_ep25:      {acc_at_ep25:.2f}%")
```

The existing optimizer init (`lr=PEAK_LR`, line 246) and the time-based schedule (`lr = PEAK_LR * progress / PCT_START` ramp and `lr = PEAK_LR * (1.0 - progress) / (1.0 - PCT_START)` decay, lines 286–292, written to every param group at lines 291–292) consume `PEAK_LR` unchanged — only the peak HEIGHT moves, the schedule SHAPE (PCT_START=0.15, anneal-to-0) is untouched.

Run cells back-to-back on the free GPU 1:
```bash
PEAK_LR=0.4 CUDA_VISIBLE_DEVICES=1 uv run train.py > run_c0.log 2>&1   # same-session baseline
PEAK_LR=0.5 CUDA_VISIBLE_DEVICES=1 uv run train.py > run_cA.log 2>&1   # primary
PEAK_LR=0.6 CUDA_VISIBLE_DEVICES=1 uv run train.py > run_cB.log 2>&1   # stress / instability probe
```
Read per cell: `grep "^best_test_acc:\|^num_epochs:\|^training_seconds:\|^total_seconds:\|^peak_lr:\|^acc_at_ep25:\|^num_params:" run_c*.log`. Pre-registered decision: **win** = a cell with best_test_acc ≥96.48 at num_epochs ≥142 AND >same-session cell-0 by ≥0.10pp, with a non-collapsed `acc_at_ep25` (≳90%, no high-LR divergence). Otherwise no-improvement. Bake the winning PEAK_LR as the static default and confirm the no-env committed run reproduces it (EXP-012 bake-and-confirm).

---

## Sources
- `train.py` (lines 20–21 PEAK_LR/BATCH_SIZE, 178 SCALE_OUT, 244–250 optimizer, 273 best_acc, 286–292 schedule, 351–357 eval/best, 373 summary) — read in full.
- `04-results.tsv` rows BASE, 001, 002, 003, 004, 008, 009, 010, 011, 012.
- `03-experiment-learnings.md`: EXP-001 base recipe (line 93–95); EXP-003 untried rider "whitening-enabled higher PEAK_LR" (line 101); noise-floor (line 32–34); EXP-009/010 Muon divergence + parity (line 66–68); EXP-012 SGD-scalars-tuned (line 60–62).
- `experiments/012/02-plan.md`: env-toggle same-session multi-cell pattern, throughput bands, thin-winner/bake-and-confirm protocol (mirrored here).
- Implicit-regularization / LR-scaling literature: Smith & Le 2018 (SGD noise scale g∝LR·N/B); Goyal et al. 2017 (linear scaling rule); Li et al. 2019 (regularization effect of large initial LR); David Page cifar10-fast / DAWNBench DavidNet (mean-loss "lambda" LR convention at batch 512).
