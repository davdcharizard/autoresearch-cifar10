# Proposal idea-03: Muon optimizer (Newton-Schulz orthogonalized momentum) for the 2D+ weights

## One-line thesis
Replace plain SGD-Nesterov on the model's ≥2D weight matrices (all conv weights + the fc
weight) with **Muon** — SGD momentum whose update matrix is orthogonalized by a fixed
Newton-Schulz quintic iteration before being applied — keeping a small SGD-momentum fallback
group for the 1D params (BN γ/β, ReZero α). Muon is the optimizer behind Keller Jordan's
newest fast-CIFAR-10 records (airbench94_muon, derived from the Muon writeup of
arXiv:2404.00498's author). The mechanism: orthogonalized updates condition the step better
and converge faster per epoch, which both (a) captures EXP-008's still-rising under-annealed
tail and (b) reaches a better fully-annealed minimum.

This is the **highest-ceiling but highest-risk** candidate for EXP-009. I am explicit below
about where it is shakier than the headline suggests, the single most important assumption
(the Muon LR), and a falsifiable first-run design.

---

## Why this targets the named limiter

The diagnosis (EXP-008 §Results, `03-experiment-learnings.md` High-Importance pattern) is that
the net is **regularization-bound with a ~4× epoch surplus**, AND that EXP-008's tail was still
mildly rising at ep150 (96.32→96.38, best==final) — a *slight under-anneal* of the
harder-augmented net. Two levers can convert that into accuracy:

1. **Capture the under-anneal headroom.** A faster-converging optimizer reaches the same loss
   level in fewer steps, so within the fixed 300s budget the low-LR tail (where "most accuracy
   gain arrives" — `03-experiment-learnings.md` Medium pattern, EXP-001) is reached *and fully
   annealed* rather than truncated. Muon's selling point in the airbench writeups is exactly
   "converges faster and reaches a lower loss in the same step budget."

2. **Reach a better minimum.** Orthogonalizing the momentum so its singular values are pushed
   toward 1 means every direction in the weight-matrix update gets a comparable-magnitude step,
   instead of SGD's update being dominated by a few large-singular-value directions. On these
   wide-shallow conv nets this is empirically a *better* minimum, not just a faster path — the
   airbench94_muon variant is what set the current speed records over the plain-SGD airbench94.

The causal chain to the metric: orthogonalized update → better-conditioned per-step descent →
(faster anneal completion) + (better generalizing minimum) → higher `best_test_acc`. This is
distinct from the EXP-005/EXP-007 capacity adds that failed: Muon does **not** add capacity or
materially cut epochs (throughput cost is small, quantified below), so it does not collide with
the recurring under-anneal-from-lost-epochs failure mode.

---

## Concrete change (files/functions in THIS codebase)

All edits are in `train.py`. Nothing else is touched. The architecture, augmentation, EMA,
whitening, TTA, schedule shape, seeds, and batch size stay byte-identical to EXP-008 so the
attribution is single-variable (optimizer only).

### Parameter inventory (verified by reading `train.py`)
- `whiten.weight` — 4D, `requires_grad=False`, already excluded by the
  `[p for p in model.parameters() if p.requires_grad]` filter. **Stays excluded.**
- conv weights in every `conv_bn` (`train.py:101-106`) — 4D `[out, in, 3, 3]`, all
  `bias=False`. → **Muon group.**
- `fc.weight` (`train.py:153`) — 2D `[10, 512]`, `bias=False`. → **Muon group.**
- BatchNorm γ and β (one pair per `conv_bn`) — 1D. → **fallback SGD group.**
- `GatedResidual.alpha` (`train.py:134`) — 1D `[1]`. → **fallback SGD group.**

So the grouping rule is simply `p.ndim >= 2` → Muon, else → fallback. (airbench94_muon filters
on `len(p.shape)==4`; that would wrongly drop our 2D `fc.weight`. Our `ndim>=2` rule correctly
sends the fc weight to Muon, which is the canonical Muon behavior for 2D matrices.)

### New code: a from-scratch Muon optimizer in `train.py`

Add near the top (torch-only, no import of any muon package — implemented from scratch as the
constraint requires):

```python
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    # Orthogonalize the 2D matrix G via the quintic Newton-Schulz iteration
    # (Keller Jordan, Muon). Pushes singular values toward 1. Runs in bf16.
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    # Muon for >=2D weights. 1D params must go to a separate (fallback) optimizer.
    def __init__(self, params, lr=0.02, momentum=0.9, nesterov=True,
                 weight_decay=5e-4, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]; mom = group["momentum"]; wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)          # decoupled-free: same as SGD coupled wd
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if group["nesterov"] else buf
                gmat = g.reshape(g.size(0), -1)     # conv [out, in*kh*kw]; fc already 2D
                update = zeropower_via_newtonschulz5(gmat, group["ns_steps"]).view_as(p)
                # RMS-match the orthogonal update to the param's fan-in so a single
                # global LR is well-scaled across matrices of different shapes.
                scale = (max(1.0, p.size(0) / gmat.size(1))) ** 0.5
                p.add_(update.to(p.dtype), alpha=-lr * scale)
```

Two scaling decisions, stated explicitly:
- **`ns_steps=5`** (the canonical Muon value, coefficients (3.4445, −4.7750, 2.0315)). I use 5
  rather than airbench94_muon's 3 because (a) 5 is the published-default that is numerically
  safest for correctness on the *first* run — convergence of the iteration is the dominant
  correctness risk and 5 iters give more margin; (b) the throughput cost of the extra 2 iters is
  negligible here (quantified below). If throughput turns out tight we can drop to 3 in a follow-up.
- **`scale = sqrt(max(1, out/in))`** is the standard Muon update-scaling that makes one global LR
  appropriate across matrices of differing aspect ratio (Keller Jordan's Muon writeup; matches
  the `len(p)**0.5/norm`-style normalization airbench94_muon applies, re-expressed as an update
  scale so we do NOT mutate `p.data`'s norm and thus do not interfere with weight-decay or EMA).
  I deliberately do **not** copy airbench94_muon's `p.data.mul_(len(p)**0.5/p.norm())` weight
  re-normalization line — that rescales the weights themselves every step and would fight both the
  whitening-independent BN scale and the AveragedModel EMA. Keeping the scale on the *update* is
  the lower-risk integration.

### Wiring into `main()` (replaces `train.py:244-250`)

```python
muon_params, sgd_params = [], []
for p in model.parameters():
    if not p.requires_grad:
        continue
    (muon_params if p.ndim >= 2 else sgd_params).append(p)

muon_opt = Muon(muon_params, lr=PEAK_LR_MUON, momentum=MOMENTUM,
                nesterov=True, weight_decay=WEIGHT_DECAY, ns_steps=5)
sgd_opt = optim.SGD(sgd_params, lr=PEAK_LR_SGD, momentum=MOMENTUM,
                    weight_decay=WEIGHT_DECAY, nesterov=True)
optimizers = [muon_opt, sgd_opt]
```

### Schedule + EMA integration (edits to the loop body, `train.py:286-311`)
- The time-based one-cycle `progress` curve is **shared**: compute the same triangular fraction
  `frac ∈ [0,1]` once, then set `g["lr"] = PEAK_LR_MUON * frac` for the Muon group's param_groups
  and `g["lr"] = PEAK_LR_SGD * frac` for the SGD group's. Concretely, replace the single
  `lr = PEAK_LR * ...` block with a `frac = (progress/PCT_START)` (ramp) or
  `(1-progress)/(1-PCT_START)` (decay) computation, then loop over *both* optimizers' param_groups
  scaling each by its own peak. This keeps both groups on the proven triangular anneal-to-~0.
- `optimizer.zero_grad(set_to_none=True)` → call on **both** optimizers (or a small helper).
- `optimizer.step()` → `muon_opt.step(); sgd_opt.step()`.
- **EMA is untouched.** `AveragedModel.update_parameters(model)` (`train.py:309`) reads the model's
  params after the step; it is optimizer-agnostic and keeps working as-is.

### New hyperparameters (top of `train.py`)
Replace `PEAK_LR = 0.4` with:
```python
PEAK_LR_MUON = 0.02   # Muon group (conv + fc weights); orthogonal ~unit-RMS updates
PEAK_LR_SGD  = 0.2    # fallback group (BN gamma/beta, ReZero alpha)
```
Justification for the concrete values is the most important part of this proposal — see next.

---

## The LR retune — the single load-bearing assumption

Muon's update is (near-)orthogonal with ~unit RMS per element after the `sqrt(out/in)` scale, so
the SGD peak of 0.4 is **wrong by roughly an order of magnitude** for the Muon group. This is the
dominant risk and cannot be swept (one run/experiment, seed fixed). I pre-register ONE pair with
the following reasoning:

- **`PEAK_LR_MUON = 0.02`.** airbench94_muon uses `lr=0.24` for its Muon group, but its
  schedule is *linear-decay-from-peak with no warmup over ~10 epochs*, its momentum is 0.6
  (vs our 0.9), it trains only ~10 epochs, and crucially its update scaling is the
  `p.norm`-renormalization variant (different effective magnitude than my update-scale variant).
  The canonical Muon default for *standard* momentum-0.9, many-epoch one-cycle setups (Keller
  Jordan's Muon writeup; nanoGPT-Muon speedruns) is ~0.02. Because our schedule (150 epochs,
  triangular warmup, momentum 0.9, EMA tail) is far closer to the canonical long-run regime than
  to airbench94_muon's 10-epoch sprint, **0.02 is the better-supported starting point than 0.24**.
  I treat 0.02 as the registered value; the falsification plan (below) reads the early trajectory
  so a gross mis-scale is detectable by ep10–25, not only at the end.
- **`PEAK_LR_SGD = 0.2`** for the 1D fallback group. BN γ/β and the ReZero α are NOT orthogonalized,
  so they want an SGD-style LR. I set it to **half** the old 0.4 rather than 0.4 because: (a) with
  the conv/fc weights now taking better-conditioned Muon steps, the BN scales see a different
  (more stable) activation distribution and a gentler BN LR reduces the risk of the
  Muon×BN interaction diverging early; (b) airbench94_muon also uses a *separate, retuned* LR for
  its non-Muon groups rather than the Muon LR. 0.2 is a single defensible value; it is the
  secondary knob and far less sensitive than the Muon LR (1D params are a tiny fraction of the
  loss curvature).

**Honesty:** the largest single failure mode is that 0.02 is itself off by 2–3× (the airbench
example pulls toward 0.24, the nanoGPT example toward 0.02 — a wide bracket). If the first run is
clearly worse than 96.38 but the early trajectory shows *stable, slow* convergence, the correct
read is "LR too low, retune up" (a second loop), not "Muon doesn't work." If it shows early
divergence/NaN, the read is "LR too high or NS numerics" (retune down). The first run is designed
to disambiguate these, not necessarily to win on the first try.

---

## Evidence it can clear >0.1pp

- **airbench94_muon is the optimizer behind the current fast-CIFAR records.** Keller Jordan's
  cifar10-airbench repo ships `airbench94_muon.py` specifically because Muon beats plain-SGD
  airbench94 at matched compute — the same wide-shallow whitened ResNet family we use
  (`knowledge/references/fast-cifar10-recipes.md` lineage; arXiv:2404.00498). The NS coefficients
  (3.4445, −4.7750, 2.0315), bf16 iteration, transpose-on-tall, and `reshape(len(g),-1)` conv
  flattening in my sketch are quoted verbatim from that file.
- **The under-anneal headroom is real and measured.** EXP-008 ended best==final at ep150 with a
  still-rising tail (96.32→96.38). A faster-converging optimizer is the textbook way to convert a
  truncated tail into a completed anneal — and "most accuracy arrives in the low-LR tail" is an
  established pattern for THIS recipe (EXP-001, Medium-importance learning).
- **Better-minimum claim has independent support:** Muon's orthogonalization is mathematically a
  steepest-descent step under the spectral norm; the published result on these nets is a *lower
  final loss at equal steps*, not merely faster early loss — i.e. a quality gain that survives to
  the annealed end, which is what the metric rewards.
- **Mechanism is throughput-compatible**, unlike the failed capacity adds (EXP-005/007) — see next.

---

## Throughput estimate (does it cut epochs → under-anneal?)

NS adds matmuls only on the weight matrices, which are tiny here:
- Largest Muon matrix: `layer3` 512→512 conv → `[512, 512*3*3] = [512, 4608]`; the
  `[256,256*9]=[256,2304]` and `[128,128*9]` ones are smaller; fc is `[10,512]`.
- Per NS iter on a `[m,n]` matrix (with the transpose making it `[min,max]`): two matmuls
  `X@X.T` (`m·m·n`) and `B@X` (`m·m·n`) ≈ `2·m²·n` FLOPs. For the 512×4608 matrix that is
  ≈ `2·512²·4608 ≈ 2.4 GFLOP` per iter, ×5 iters ×2 (the A@A term) ≈ **~24 GFLOP/step** worst-case,
  summed over all ~10 Muon matrices maybe **~40–60 GFLOP/step** total.
- A single forward+backward of this net on a 512-image batch is on the order of **hundreds of
  GFLOP** (≈3.6 GFLOP/img fwd → ~5–6 TFLOP fwd+bwd per batch). So NS is **<1–2% of step FLOPs**,
  and these are small dense bf16 matmuls the GPU runs at high efficiency.
- **Estimate: <2% step-time overhead → ≤2–3 fewer epochs (≈147–150 vs 150).** Well inside the
  142–150 "normal band" (`03-experiment-learnings.md`) — this is NOT an under-anneal risk like the
  EXP-007 widen (150→94). I will still read `num_epochs` as the first-class confound check.

If measured overhead is larger than expected, dropping `ns_steps` 5→3 (airbench94_muon's value)
roughly halves it with little accuracy cost — a cheap follow-up knob.

---

## Verification / falsification (pre-registered diagnostics)

Run once on GPU 1 (`CUDA_VISIBLE_DEVICES=1`) under `timeout 600`, full 300s training budget,
seeds `torch.manual_seed(42)/torch.cuda.manual_seed(42)` UNCHANGED.

- **Correctness smoke (before the full run):** assert `zeropower_via_newtonschulz5` on a random
  `[512,4608]` and a tall `[4608,512]` returns finite values with singular values within ~[0.7,1.3]
  of 1 (the NS quintic does not converge to exactly 1 in 5 iters — that band is expected and fine);
  assert the optimizer step runs and `num_params==7,784,627` is unchanged (optimizer-only change).
- **Divergence guard:** the run must not NaN/inf. A diverging Muon LR shows as exploding
  `smooth_train_loss` in the first ~hundred steps — visible in the existing step printout.
- **Primary success:** `best_test_acc ≥ 96.48%` (the EXP-009 bar) AND clearly above the ~0.1pp
  noise floor; `num_epochs` in [142,150]; `total_seconds` ≈ 440–450s (no throughput regression).
- **Trajectory read (the informative part even if it loses):**
  - *Muon working:* early epochs at-or-above EXP-008's (ep25 ≳ 92.3%), tail fully anneals (NOT
    best==final still-rising) and finishes higher.
  - *LR too low:* stable but *slower* trajectory than EXP-008 (ep25 well below 92.3, e.g. ~90%),
    monotone, finishing below baseline → register "retune PEAK_LR_MUON up to ~0.04–0.05" for loop 2.
  - *LR too high / NS unstable:* early loss spikes, jagged accuracy, possible NaN → "retune down to
    ~0.01" for loop 2.
- **Falsification:** if the trajectory is *stable and well-converged* (tail fully annealed, ep25 in
  band) yet `best_test_acc < 96.38`, that falsifies the "better-minimum at this scale" claim — Muon
  helped convergence but not generalization here, and the lever is exhausted for this net.

---

## Strongest risk (ranked)

1. **Muon LR mis-scaled (dominant).** One run, fixed seed, no sweep → if 0.02 is off by >~2× the
   first run likely lands below 96.38. Mitigation: the pre-registered trajectory read turns a
   "loss" into a *directional* retune for loop 2, so the experiment is informative either way. Be
   honest that this idea plausibly needs **2 loops** (stabilize+coarse-tune, then fine-tune).
2. **NS numerics under bf16.** The quintic can mildly amplify if the input norm is tiny or the
   matrix is rank-deficient; the `X/(X.norm()+eps)` normalization and 5-iter margin guard this, but
   a bug in transpose handling or the conv `reshape(len(g),-1)` could silently degrade updates
   without NaN-ing. Mitigation: the singular-value smoke test catches a broken iteration.
3. **Muon×BN / EMA interaction.** Orthogonalized weight steps change the activation statistics BN
   sees; combined with the EMA tail this could under- or over-shoot. Mitigation: keeping the update
   scale OFF the weight norm (not copying airbench94_muon's `p.norm` renorm) and using a gentler
   `PEAK_LR_SGD=0.2` reduces this; the EMA path is unchanged and optimizer-agnostic.
4. **Coupled weight decay on Muon updates.** I fold wd into the grad before momentum (matching the
   existing SGD's coupled wd) rather than decoupled — lowest-surprise choice for single-variable
   attribution; decoupled wd is a separate future lever, not bundled here.

---

## Effort

**Medium–high** for the first loop (new optimizer class + NS function + dual-optimizer wiring +
shared-schedule edit; all localized to `train.py`, ~50 lines). Realistically **budget 2 loops**:
loop 1 stabilizes the implementation and coarse-locates the Muon LR via the trajectory read; loop 2
fine-tunes `(PEAK_LR_MUON, PEAK_LR_SGD)` and optionally `ns_steps`. The first run is designed to be
maximally informative regardless of whether it clears the bar.

---

## Honest bottom line
This is the candidate with the highest ceiling (it is *the* lever behind the records above ours)
and the highest variance. The mechanism is well-matched to the named limiter (it directly attacks
the under-annealed tail + better-minimum), the throughput cost is genuinely small (unlike the
failed capacity adds), and the implementation is fully torch-only and localized. The honest caveat
is the LR: with a fixed seed and one run, the first attempt is as much a *calibration* of
`PEAK_LR_MUON` as a bid to beat 96.38, so success should be judged on the trajectory diagnostics as
well as the final number, and a second tuning loop is likely.

## Sources
- Keller Jordan, cifar10-airbench `airbench94_muon.py` (NS coefficients, bf16 iteration, conv
  flattening, param grouping, separate group LRs): https://github.com/KellerJordan/cifar10-airbench
- airbench paper, arXiv:2404.00498 (whitened wide-shallow ResNet family; record lineage).
- `03-experiment-learnings.md` (noise floor, under-anneal failure mode, low-LR-tail pattern),
  `experiments/008/04-analysis.md` (still-rising tail headroom),
  `knowledge/references/fast-cifar10-recipes.md` (recipe lineage).
