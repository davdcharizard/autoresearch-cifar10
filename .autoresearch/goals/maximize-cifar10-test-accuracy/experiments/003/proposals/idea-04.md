# Idea-04 (moonshot): Muon optimizer for conv/Linear weight matrices

**Goal**: maximize `best_test_acc` (%) within the fixed 300s training-time budget.
**Baseline**: 95.72% (EXP-002). **Bar**: ≥ +0.1pp ⇒ **≥95.82%**.
**Effort**: **HIGH** (custom optimizer, fiddly LR/scaling tuning, real divergence risk).

---

## 1. Limiter targeted

From the diagnosis and `03-experiment-learnings.md`:
- EXP-001 established the DavidNet + time-based one-cycle recipe (95.22%) and found **most
  accuracy arrives in the low-LR tail of a completing one-cycle** (Patterns §Medium).
- EXP-002 added EMA + flip-TTA (+0.50pp → 95.72%), explicitly **eval-side** and orthogonal to
  the training trajectory. It did *not* touch the optimizer.

So the **training-dynamics optimizer is still the stock `optim.SGD(lr=0.4, momentum=0.9,
wd=5e-4, nesterov=True)`** (`train.py:167-173`). The named limiter this idea attacks is
**optimization efficiency per step / per epoch**: with ~183 epochs available in 300s, the
question is whether a better-conditioned update can reach a lower-loss / flatter basin within
the same wall-clock, lifting the tail accuracy that EMA+TTA then denoises. Muon conditions the
update *spectrum* (replaces the raw momentum matrix `G` with its nearest semi-orthogonal matrix
`UVᵀ`), so every singular direction of each weight matrix receives a comparably-sized update
instead of the SGD update being dominated by a few large singular directions. In the
epoch-limited fast-CIFAR regime this is exactly the lever airbench used to set its single-GPU
record.

---

## 2. Mechanism (causal chain, not "should help")

1. SGD-momentum produces an update matrix `G` (the momentum buffer) whose singular value
   spectrum is highly anisotropic. A scalar LR must be sized for the *largest* singular
   direction, so smaller directions are under-trained per step.
2. Muon replaces `G` with `zeropower(G) ≈ U Vᵀ` (all singular values ≈ 1) via ~3-5
   Newton-Schulz quintic steps. Every direction now gets a unit-scale step; the effective LR is
   decoupled from the gradient's spectral norm.
3. With a uniform update spectrum, the same number of epochs covers more useful descent
   directions per step → lower train loss / flatter minimum at the *same* training-time budget.
4. The improved iterate feeds the **unchanged** EMA+TTA eval path (EXP-002), so any
   training-side gain is preserved and denoised in the tail, where EXP-001 showed gains
   concentrate.
5. Net: a measurable lift in `best_test_acc` *if* the Muon LR is set correctly and the per-step
   Newton-Schulz overhead does not erase too many epochs.

This is precisely the path airbench took from its SGD recipe to `airbench94_muon.py`, which
holds the current single-GPU CIFAR speed record.

---

## 3. Concrete change to `train.py`

All edits are in `train.py` only (`prepare.py` frozen). Architecture (`ResNet9`), Cutout, label
smoothing, EMA, TTA, weight decay, and the time-based one-cycle schedule are **unchanged** for
attribution. Only the optimizer construction and the per-step update are replaced.

### 3a. Newton-Schulz routine + Muon optimizer (pure torch, no new deps)

Add near the top of `train.py` (after imports, before `Cutout`). This is the verbatim airbench
algorithm (Keller Jordan), adapted to run **without `@torch.compile`** (eager is fine — the NS
iteration is 3 small matmuls per weight; our codebase does not use `torch.compile` anywhere):

```python
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    # Orthogonalize G (~nearest semi-orthogonal matrix) via a quintic Newton-Schulz iter.
    # Coeffs from Keller Jordan's Muon; G must be 2D. Runs in bf16 for speed/stability.
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)          # scale so all singular values land in [0, 1]
    transposed = X.size(0) > X.size(1)
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
    """Muon for 2D+ weight matrices. 1D params (BN, bias) must NOT be in here."""
    def __init__(self, params, lr=1e-3, momentum=0.9, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, nesterov, ns = (group["lr"], group["momentum"],
                                     group["nesterov"], group["ns_steps"])
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if nesterov else buf
                g2d = g.reshape(g.shape[0], -1)
                update = zeropower_via_newtonschulz5(g2d, steps=ns).view(g.shape)
                # aspect-ratio scale so the update RMS is ~weight-init scale (see §3c)
                fan_out, fan_in = g2d.shape
                scale = (max(1.0, fan_out / fan_in)) ** 0.5
                p.add_(update.to(p.dtype), alpha=-lr * scale)
```

**Decision — scaling (the load-bearing choice).** airbench's `step()` instead does
`p.data.mul_(len(p.data)**0.5 / p.data.norm())` (a per-step weight-norm *constraint*) and adds
the raw orthogonalized update with no aspect-ratio factor. That weight-renorm is tightly coupled
to airbench's lack of weight decay and its specific lr=0.24. Porting it into *our* recipe would
collide with `WEIGHT_DECAY=5e-4` and the `scale_out=0.125` logit scaling, confounding
attribution. I therefore use the **standard Muon update form** (recent Muon writeups):
orthogonalized update times an aspect-ratio factor `max(1, fan_out/fan_in)**0.5`, applied as a
normal additive step, and **keep our SGD-style weight decay on the Muon group** by leaving
`weight_decay` in the SGD-handled groups only (see §3b) OR by adding a decoupled
`p.mul_(1 - lr*wd)` line. Recommendation: start with **no weight decay on Muon params** (Muon's
orthogonalization already bounds update scale; airbench runs Muon params WD-free), keep WD only
on the SGD group. This is the single biggest tuning knob and the main divergence risk.

### 3b. Param grouping (replace the single SGD at `train.py:167-173`)

The standard Muon recipe excludes 1D params, biases, and (often) the head/input from
orthogonalization. Our net (`ResNet9`) is bias-free in every conv (`bias=False`) and the Linear
(`fc` `bias=False`), so the only non-Muon params are the **BatchNorm weight+bias** (1D). Decision
on the **final Linear head**: airbench keeps the head on SGD with its own LR. Our `fc` is
`512×10` (tiny, very wide aspect ratio) — orthogonalizing a 10×512 matrix is degenerate
(rank ≤ 10). **Put `fc` on SGD, not Muon.** The first conv (`prep`, 64×27) is full-rank and fine
for Muon.

```python
muon_params, sgd_params = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim >= 2 and "fc" not in name:   # conv weights only
        muon_params.append(p)
    else:                                  # BN weight/bias (1D) + fc head
        sgd_params.append(p)

muon = Muon(muon_params, lr=MUON_LR, momentum=MOMENTUM, nesterov=True, ns_steps=NS_STEPS)
sgd = optim.SGD(sgd_params, lr=PEAK_LR, momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY, nesterov=True)
optimizers = [muon, sgd]
```

New constants near the other hyperparameters (`train.py:19-31`):
```python
MUON_LR = 0.05     # peak Muon LR (orthogonal-update scale; see §3c). MAIN TUNING RISK.
NS_STEPS = 5       # Newton-Schulz iterations (airbench uses 3; 5 is the canonical default)
```

### 3c. LR handling — how Muon LR relates to the existing one-cycle (`train.py:209-215`)

The existing loop computes a single scalar `lr` from `total_training_time/TIME_BUDGET_S` and
writes it into `optimizer.param_groups`. Muon's update has **~unit spectral scale** (orthogonal
matrix), so the SGD peak `0.4` is the wrong magnitude — it would be a huge step. Reasoning for
the Muon peak:
- An orthogonal `d_out×d_in` update has Frobenius norm `√min(d_out,d_in)` ≈ RMS entry
  `1/√max(d_out,d_in)`. The aspect scale `max(1,fan_out/fan_in)**0.5` makes the per-element RMS
  ≈ `1/√fan_in`, i.e. comparable to a Kaiming-init weight. So a Muon LR of order **0.02–0.1**
  moves each weight by a few percent of its init scale per step — the right ballpark.
- airbench uses Muon lr=0.24 with a *3-epoch-fast* whitening front-end and no WD; for our
  183-epoch full-WD recipe a smaller peak (~0.05) is the principled starting point.

Schedule both optimizers off the **same** time-based one-cycle shape, but with **separate
peaks** — keep `PEAK_LR=0.4` for the SGD group (BN+head, where the SGD spectral argument still
holds) and `MUON_LR=0.05` for the Muon group:

```python
progress = min(1.0, total_training_time / TIME_BUDGET_S)
if progress < PCT_START:
    frac = progress / PCT_START
else:
    frac = (1.0 - progress) / (1.0 - PCT_START)
for g in muon.param_groups:
    g["lr"] = MUON_LR * frac
for g in sgd.param_groups:
    g["lr"] = PEAK_LR * frac
```

### 3d. Step both optimizers (replace `train.py:222-227`)

```python
for opt in optimizers:
    opt.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
loss.backward()
for opt in optimizers:
    opt.step()
```

EMA (`ema_model.update_parameters` at `train.py:231-233`), TTA gating, eval path
(`train.py:264-272`), and the summary are all **untouched**. A planner can produce this as a
contained diff: one inserted block (NS+Muon), one rewritten optimizer-construction block, one
rewritten LR block, one rewritten step block.

---

## 4. Evidence

- **airbench Muon is the current single-GPU CIFAR record** (`knowledge/references/
  fast-cifar10-recipes.md` §lineage; `airbench94_muon.py`). The verbatim reference
  implementation (zeropower coeffs `(3.4445, -4.7750, 2.0315)`, bf16, Frobenius pre-scale,
  transpose-on-tall, `g.reshape(len(g),-1)` for conv) is reproduced in §3a from the repo source,
  so the algorithm is exactly pinned — no guessing.
- **Muon writeup (Keller Jordan, 2024/2025)**: Muon applies to 2D hidden / conv weights;
  scalars, vectors, input/output layers excluded — matching our grouping (BN 1D + `fc` head on
  SGD). 5 NS steps suffice for small conv nets. Nesterov-style momentum empirically best.
- **Code facts that make integration safe**: `ResNet9` is fully bias-free
  (`train.py:72,97` — `bias=False`), so the only non-Muon weights are BN 1D params and the head
  — a clean partition. The loop already writes `lr` into `param_groups` each step
  (`train.py:214-215`), so swapping in two optimizers with separate peaks is a localized change.
- **Orthogonality to prior wins**: EXP-002 EMA+TTA acts on the evaluated iterate
  (`03-experiment-learnings.md` Patterns §High) and is untouched here, so a training-side Muon
  gain composes additively on top of 95.72% rather than replacing it.

---

## 5. Strongest risk / assumption that most needs to hold

**The Muon peak LR must be in the right band, and it is genuinely unknown for this 183-epoch,
full-WD, `scale_out=0.125` recipe.** This is the dominant failure mode:
- Too high → divergence (orthogonal updates are unit-scale; a 0.4-style LR would explode). The
  bf16 NS iteration plus a mis-scaled step is a real NaN risk.
- Too low → under-stepping; Muon contributes less descent than the tuned Nesterov-SGD it
  replaces, and the per-step NS overhead costs epochs → **net negative**.
- The aspect-ratio scaling vs airbench's weight-renorm choice (§3a) is a second coupled unknown;
  picking the wrong normalization silently mis-sizes every update.

Secondary risks:
- **Per-step NS overhead reduces epoch count.** 3-5 matmuls per conv weight per step, eager (no
  `torch.compile` in our stack). The largest matrices are 512×(512·9)=512×4608 and
  256×2304 — `X@X.T` is 512×512, cheap relative to the conv forward/backward, but summed over
  ~9 conv layers × ~17k steps it is non-trivial. Honest expectation: a few-to-~15% epoch drop
  (cf. EXP-002's 192→183 from much lighter EMA overhead). If NS dominates, `ns_steps=3` (airbench
  value) is the first lever.
- **The marginal-gain regime is unfavorable.** Muon's biggest reported wins are epoch-starved /
  large-model. At 183 epochs on a 6.5M net against an *already well-tuned* one-cycle
  Nesterov-SGD, the headroom may be small or zero even with a correct LR.

---

## 6. Honest expected-magnitude estimate vs the 95.82% bar

This is a **moonshot, plausibly net-zero-to-negative**, and I will not inflate it:

- **Most likely (~50%)**: lands within ±0.15pp of 95.72%, i.e. **fails to clear 95.82%**. The
  base recipe is already well-tuned; Muon's structural advantage is muted at 183 epochs, and
  the epoch loss from NS overhead roughly cancels any conditioning gain.
- **Upside (~25%)**: with a well-chosen `MUON_LR` (~0.03-0.08), the better-conditioned updates
  buy a flatter tail and **+0.1 to +0.4pp → 95.82-96.1%**, clearing the bar. This is the
  airbench-style outcome and the reason to try it.
- **Downside (~25%)**: LR mis-set or normalization wrong → divergence / NaN, or clear
  under-stepping → **below 95.6%**, a `no-improvement`.

Given the HIGH effort and the LR-tuning risk, this is a higher-variance bet than the
EMA-decay/whitening incremental ideas. Its value is the asymmetric upside (matching the
record-setting optimizer) and that, win or lose, it cleanly answers whether the *optimizer* is a
remaining lever on this recipe.

**Fallback ladder if the first run misbehaves** (each is a one-line/one-constant change, no
re-architecture):
1. Diverges/NaN → halve `MUON_LR` (0.05 → 0.025 → 0.0125).
2. Under-steps (loss tracks SGD but never beats it) → raise `MUON_LR` toward 0.1-0.15.
3. Epoch count drops too far → `NS_STEPS=3` (airbench default).
4. Net-negative after LR is dialed in → **revert to the EXP-002 baseline optimizer** and record
   "optimizer is not the binding limiter on this recipe" as the learning. The EMA+TTA eval path
   is unchanged throughout, so the 95.72% machinery is intact for the revert.

A clean within-budget run that simply under-performs is still an informative `no-improvement`:
it pins whether the stock Nesterov-SGD one-cycle is already near-optimal for DavidNet at 300s.
