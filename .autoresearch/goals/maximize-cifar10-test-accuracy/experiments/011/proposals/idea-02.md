# Proposal (EXP-011, idea-02): Recipe-scalar refresh — weight-decay SHAPING + label-smoothing retune

## One-line summary
The core SGD scalars `WEIGHT_DECAY=5e-4` and `LABEL_SMOOTHING=0.2` were tuned for the EARLY recipe (EXP-001/002, ~95.2-95.7%) and were never revisited after EXP-008's +0.38pp augmentation change (cutout 8→12 + light RandomErasing) lifted the net to 96.38%. Two zero-throughput-cost refreshes: (a) **weight-decay shaping** — split the optimizer into param groups so L2 decay applies only to conv/fc weight matrices and is removed (wd=0) for BN γ/β and the ReZero α scalar; (b) **label-smoothing retune** — drop LS 0.2→0.1, since the now-stronger input augmentation already supplies the regularization that the heavy LS was compensating for. Both are throughput-free regularization-shaping levers, the only lever class with a proven win at this saturated operating point (EXP-008).

## Mechanism (tied to the named limiter)

**Named limiter (from diagnosis):** the net is **regularization-bound / saturated with a ~4× epoch surplus** — it fits ~150 epochs in 300s for ~96% vs airbench96's ~37 for the same accuracy. The optimizer axis is exhausted (EXP-009/010: tuned Muon only ties SGD), and capacity adds under-anneal (EXP-005/007). The productive lever class is **throughput-free regularization shaping** — EXP-008 (+0.38pp) is the existence proof. These two scalars are exactly that class and are demonstrably stale.

**Why "stale" is the right framing, concretely.** `WEIGHT_DECAY=5e-4` and `LABEL_SMOOTHING=0.2` entered the recipe at EXP-001 (the DavidNet/johanwind defaults) when the augmentation was only `Cutout(8)` and the net topped out at 95.22%. Since then the regularization budget of the recipe has changed substantially: EXP-002 added EMA (a strong implicit regularizer), EXP-008 raised Cutout 8→12 AND added RandomErasing(p=0.25). Total input-space + weight-space regularization is now materially higher than when 5e-4/0.2 were chosen, yet neither scalar was re-derived. On a regularization-bound net, the *total* regularization dose and its *allocation* across mechanisms is precisely what sets the fully-annealed generalization ceiling — so a stale allocation is a credible source of left-on-the-table accuracy.

Two distinct sub-mechanisms:

**(a) Weight-decay shaping.** Currently `weight_decay=5e-4` is applied uniformly to every learnable parameter (`train.py:244-250`). This is wrong in two specific places:
- **BN γ/β.** Decaying a BatchNorm scale γ toward 0 shrinks post-BN activation magnitude, but BN already controls activation scale by construction — so decay on γ is a spurious per-channel-gain bias, not real complexity control. Removing it is the "Bag of Tricks" / fastai / timm convention. **Honest caveat (per the idea brief):** because BN makes the preceding conv weight scale-invariant (scaling a pre-BN conv weight leaves the block output unchanged), the dominant real effect of conv-weight decay in a BN net is on the *effective learning rate* (it keeps ‖weight‖ from growing, which keeps the effective LR from shrinking), not on the loss landscape's complexity term directly. So WD-shaping is partly an effective-LR knob, and the literature returns are **mixed/diminishing** (He et al. 2019 found diminished returns from the bundle; some ResNet-18 studies found targeted norm/bias decay changes worth ~0.8pp, others near-zero). Treat this as a modest-confidence tunable, not a guaranteed win.
- **ReZero α.** `GatedResidual.alpha` (`train.py:134`, shape [1], init 0) is the gate that ramps EXP-004's layer2 block from identity (+0.13pp). Uniform `weight_decay=5e-4` applies a constant restoring force toward 0 on α — i.e. it actively fights the capacity ramp. EXP-004 measured α.grad ≈ 0.0179, so the `lr·wd·α` decay term is non-negligible relative to the data gradient. Excluding α lets the gate reach a larger steady state. This sub-mechanism is specific to our net and not priced into the generic Bag-of-Tricks accounting.

**(b) Label-smoothing retune (0.2→0.1).** LS=0.2 is on the high end (the fast-CIFAR lineage uses 0.1-0.2; airbench/hlb use ~0.1, DavidNet ~0.2). LS regularizes by softening targets, capping confident logits. With EXP-008's stronger input augmentation now providing more regularization, the heavy LS may now be *over*-regularizing — flattening the target distribution more than the harder-augmented data needs, which can cost a few tenths at the annealed optimum. Dropping to 0.1 reduces the target-side regularization to re-balance against the increased input-side regularization. This directly tests the "stale scalar" hypothesis on the loss side.

## Concrete change (this codebase)

Two edit sites in `main()`, both verified against the actual model definition (`train.py:101-185`), not filenames.

### Site 1 — optimizer param-group partition (`train.py:244-250`)

Current:
```python
optimizer = optim.SGD(
    [p for p in model.parameters() if p.requires_grad],  # exclude frozen whitening conv
    lr=PEAK_LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True,
)
```

Parameter inventory I confirmed by reading the model (`train.py`):
- `whiten.weight` — 4-D, `requires_grad=False` (set `train.py:147,169`) → must stay excluded; the partition filters on `p.requires_grad` first.
- All `conv_bn` Conv2d weights (`train.py:103`, `bias=False`) — 4-D → **decay group**.
- All `BatchNorm2d` (`train.py:104`) γ and β — both 1-D → **no-decay group**.
- `GatedResidual.alpha` (`train.py:134`, `torch.zeros(1)`) — shape [1], `ndim==1` → **no-decay group**.
- `fc` (`train.py:153`, `nn.Linear(512, 10, bias=False)`) — weight 2-D → **decay group**.
- There are **no bias parameters anywhere** (every conv and the fc are `bias=False`), so the only 1-D learnable params are BN γ/β and α. A `p.ndim <= 1` split is exact and robust here — verified against the full module list.

Replacement:
```python
decay_params, no_decay_params = [], []
for p in model.parameters():
    if not p.requires_grad:          # skip frozen whitening conv
        continue
    if p.ndim <= 1:                  # BN γ/β and ReZero α scalar
        no_decay_params.append(p)
    else:                            # conv (4-D) and fc (2-D) weight matrices
        decay_params.append(p)

# sanity (cheap, keep): no-decay group is small (BN γ/β + α only), not stray conv
assert no_decay_params and sum(p.numel() for p in no_decay_params) < 0.05 * learnable_params

optimizer = optim.SGD(
    [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=PEAK_LR,
    momentum=MOMENTUM,
    nesterov=True,
)
```

Planner notes (verified in code):
- The LR-schedule loop at `train.py:291-292` already does `for g in optimizer.param_groups: g["lr"] = lr` — it writes LR to **every** group, so both groups receive the identical one-cycle LR with no change needed. This is the only other place `param_groups` is touched.
- `momentum`/`nesterov` are optimizer-level defaults applying to both groups; only `weight_decay` is overridden per-group. Standard PyTorch idiom.
- `learnable_params` is already computed at `train.py:241` — reuse it for the assert.

### Site 2 — label smoothing (`train.py:24`)

Change the constant `LABEL_SMOOTHING = 0.2` → `0.1`. It is consumed once at `train.py:251` (`criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)`). No other change needed.

## Recommended config + defaults — what to change vs hold

This is the crux. The honest tension: each knob individually is likely **sub-noise (~0.1pp floor)**, so a single-knob A/B may be unprovable; but bundling raises attribution ambiguity. My recommendation, ordered by what I'd actually run:

**Primary recommendation — run as a SMALL 3-cell mini-sweep (compressed), not a single A/B.** Given the ~0.1pp noise floor and that each lever is plausibly worth only 0.05-0.15pp, a single full-budget A/B per knob is a poor use of the loop. Run three full-budget configs and read the *ranking + magnitude* jointly:

| cell | WD shaping (BN/α → wd=0) | conv/fc WD | LS | rationale |
|------|---|---|---|---|
| **A (high-confidence first cut)** | yes | 5e-4 (hold) | 0.2 (hold) | isolated WD-shaping; the "standard practice, never applied" lever |
| **B (loss-side refresh)** | yes | 5e-4 (hold) | **0.1** | shaping + the LS retune, the most likely-stale scalar |
| **C (combined recipe refresh)** | yes | 5e-4 (hold) | **0.1** + (optional) conv/fc WD → **8e-4** | with γ/β/α decoupled, the conv-only WD can often be raised; tests whether the freed budget wants re-allocating to where decay does real work |

Pre-register the read: B vs baseline is the headline test (loss-side retune is the highest-prior-probability stale scalar); A isolates whether WD-shaping alone moves anything; C probes the joint optimum. If the loop only affords ONE full run, run **B** (shaping + LS 0.1) — it bundles the two highest-confidence stale-scalar fixes, and if it clears the bar the follow-up A-vs-B decomposition is cheap.

**Do NOT** retune PEAK_LR in this experiment (see below) and do NOT touch augmentation/EMA/architecture — hold everything else byte-identical to EXP-008 for clean attribution.

**Why not just one minimal single-variable A/B?** Because the established noise-floor protocol says a sub-0.1pp single-run delta is unprovable, and each knob alone is plausibly sub-0.1pp. A 2-3 cell sweep that shares the EXP-008 baseline as the control lets us read a *consistent* direction across cells (e.g. all three above baseline) as evidence even when no single cell clears 0.1pp decisively — and gives the decomposition needed to attribute any win. This is the honest best execution; I flag it explicitly rather than pretending a single A/B is adequate.

## Effective-LR side effect — hold PEAK_LR fixed

Removing WD on the conv weights changes the effective LR (BN scale-invariance: with no decay, ‖conv weight‖ drifts upward over training, which *lowers* the effective LR in the late schedule). This could in principle want a compensating PEAK_LR nudge. **Recommendation: hold PEAK_LR=0.4 fixed.** Rationale:
- Co-tuning PEAK_LR confounds the attribution and forfeits the clean read against the EXP-008 baseline.
- The WD on the conv/fc weights is *unchanged* (still 5e-4) in cells A/B — only γ/β/α decay is removed. The conv-weight effective-LR drift is therefore driven by the *unchanged* 5e-4, exactly as in the baseline; removing BN-γ decay does not directly change the conv-weight norm dynamics (γ decay does not act on conv weights). So the effective-LR perturbation in A/B is second-order, not first-order — holding PEAK_LR is safe and keeps attribution clean.
- If C raises conv WD to 8e-4, that *does* change the conv effective-LR meaningfully; that is why C is the lowest-priority/most-confounded cell. Keep PEAK_LR fixed even there for one-variable-at-a-time discipline, and read C cautiously.

## Expected effect (pp + reasoning) — honest, likely small

- **Published / prior effect sizes.** No-decay-on-BN/bias is a reliable-but-small Bag-of-Tricks lever, isolated effect typically ~0.05-0.2pp on small CIFAR ResNets, with **mixed/diminishing** returns reported (He 2019). LS retunes of 0.1↔0.2 on already-regularized CIFAR nets are typically ~0.05-0.15pp. The α-decoupling is a net-specific extra few hundredths.
- **Saturated-net discount.** This net is already at 96.38% with LS 0.2 + Cutout12 + RandomErasing + EMA + wd 5e-4. The marginal value of any single regularization correction shrinks near saturation. That pushes each individual knob toward the **low end**.
- **Realistic point estimate.** Each knob: ~0.05-0.15pp, individually possibly within noise. The *bundle* (cell B): plausibly **0.1-0.25pp** if the stale-scalar hypothesis is right and the effects are roughly additive — i.e. a credible but not confident clearing of the +0.10pp bar.
- **Asymmetry of downside (the real selling point).** All cells cost zero epochs (param-grouping and an LS constant change nothing in the forward/backward compute), so unlike EXP-005/007 they **cannot lose by under-annealing**. Expected `num_epochs` ~142-150. The worst realistic case is "indistinguishable from baseline," not a regression. This favorable downside is why the lever is worth a slot despite modest upside.

## Pre-registered success / failure read

- **Throughput guard (attribution):** confirm `num_epochs ∈ ~[140,155]` and `total_seconds ≈ 440-450s` for every cell — rules out the epoch-count confound. If `num_epochs` is depressed by shared-host contention (cf. EXP-010), the absolute numbers are throughput-confounded and only the cross-cell *ranking* is valid (all cells equally slowed).
- **Success:** any cell with `best_test_acc ≥ 96.48%` (+0.10pp) AND clearly above the ~0.1pp noise floor, with normal epoch count. Promote the winning config; decompose if it was bundled.
- **Soft-positive (consistent direction):** all cells land in [96.40, 96.48] above baseline with normal epochs → directionally encouraging but not a pass against the noise floor; record as no-improvement, note the consistent direction, and consider a confirmation. A single +0.07pp cell is NOT a pass (noise-floor protocol).
- **Null:** cells scatter around [96.30, 96.42] with normal epochs → the stale-scalar hypothesis does not clear the bar on this saturated net; record no-improvement and conclude the scalars, while theoretically mis-set, do not move the annealed optimum enough to matter.
- **Surprise regression** below ~96.20 with normal epochs → the removed γ/β/α decay or the lower LS was load-bearing regularization (unlikely given the rich aug); worth flagging.

## Strongest risk
**Each change is individually sub-noise on this saturated, heavily-regularized net, making a clean attribution hard.** The assumption that most needs to hold is that the staleness is real AND that fixing it shifts the fully-annealed optimum by **more than the ~0.1pp floor** — at 96.38 with LS 0.2 + Cutout12 + RandomErasing + EMA, the marginal regularization correction may be worth only a few hundredths per knob. Bundling (cell B/C) raises the expected magnitude but muddies which knob did the work. The mini-sweep design (shared baseline, cross-cell ranking, cheap A-vs-B decomposition on a win) is the mitigation, but the honest assessment is this is a **modest-confidence, low-downside** lever, not a likely decisive win. The WD-shaping half in particular is, per the literature, a mixed/diminishing-return effective-LR knob rather than a guaranteed regularizer.

## Effort
**Low.** Two small edits (one ~10-line optimizer partition at `train.py:244-250`, one constant at `train.py:24`), no new deps, no schedule/architecture change, zero throughput cost. As a single run (cell B): one 300s run + smoke. As the recommended mini-sweep: 2-3 full 300s runs (B is the must-run; A and C are the decomposition/joint-optimum cells) ≈ 25-35 min total including the off-budget startup. Lowest implementation cost and lowest downside risk in the candidate set; its weakness is upside uncertainty.

## Evidence pointers
- **Bag of Tricks for Image Classification with CNNs**, He et al., CVPR 2019, **arXiv:1812.01187** — §4 "No bias decay" (exclude BN γ/β + biases from weight decay) and label-smoothing discussion; the convention this proposal implements. Note: He et al. report these as a *bundle* with diminishing isolated returns — basis for the modest-confidence framing. Cited in `knowledge/references/fast-cifar10-recipes.md:27`.
- **fast-CIFAR lineage** (`knowledge/references/fast-cifar10-recipes.md:13`): airbench96/hlb use cutout=12 and LS ~0.1; DavidNet/johanwind use LS up to 0.2 with wd 5e-4 — establishes 0.2 as the high end and motivates the 0.2→0.1 retune now that aug matches airbench's cutout=12.
- **EXP-008** (`experiments/008/04-analysis.md`): the augmentation change that lifted 96.00→96.38 and made these scalars stale; its "Unexplored Avenues"/"Next Steps" explicitly list "compose decoupled weight decay (idea-01) and/or a label-smoothing retune" as the recommended throughput-free follow-up (lines 33, 39).
- **EXP-008 idea-01** (`experiments/008/proposals/idea-01.md`): the decoupled-WD proposal, never executed (EXP-008 chose augmentation); this proposal extends it with the LS retune and the post-EXP-008 stale-scalar framing, reusing its verified parameter inventory.
- **EXP-004** (`experiments/004/04-analysis.md`): ReZero α delivered +0.13pp with measured α.grad ≈ 0.0179 ≠ 0 — basis for the α-decoupling sub-mechanism (decay term non-negligible vs the gate's data gradient).
- **EXP-009/010** (`03-experiment-learnings.md` Low-Importance Muon entries; `experiments/010/04-analysis.md`): optimizer axis exhausted (Muon ties SGD) → "pursue throughput-free regularization levers instead" — directly motivates this lever class.
- **Noise-floor protocol** (`03-experiment-learnings.md` High-Importance): ~0.1pp floor and "treat sub-0.1pp single-run wins as unproven" — drives the mini-sweep design and the pre-registered reads.
- **EXP-001 learning** (`03-experiment-learnings.md`, "most accuracy gain arrives in the low-LR tail"): the mechanism by which a correctly re-allocated penalty acts on the fully-annealed optimum.

## Out-of-scope confirmation
Edits only `train.py` (`train.py:24` and `train.py:244-250`); `prepare.py` untouched; no new dependencies (torch only); seeds `torch.manual_seed(42)`/`torch.cuda.manual_seed(42)` unchanged (`train.py:199-200`); ≤1 eval/epoch unchanged (single `evaluator.evaluate` site, `train.py:349`); runs on GPU 1 (`CUDA_VISIBLE_DEVICES=1`); 300s training budget and time-based schedule unchanged.
