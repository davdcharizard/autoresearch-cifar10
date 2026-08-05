# Proposal (EXP-008, idea-01): Decoupled weight decay — exclude 1-D params (BN γ/β, ReZero α) from weight decay

## One-line summary
Split the SGD optimizer into two param groups — `weight_decay=5e-4` for multi-dim weights (conv 4-D, fc 2-D) and `weight_decay=0.0` for 1-D params (BatchNorm affine γ/β, the ReZero `alpha` scalar) — a zero-throughput-cost regularization-quality lever that is standard practice ("Bag of Tricks", He et al. CVPR 2019, arXiv:1812.01187) but is currently NOT applied: `weight_decay=5e-4` is applied uniformly to every learnable parameter.

## Mechanism (tied to the named limiter)

The diagnosis says the net is **saturated / regularization-bound**: it fits ~142–150 epochs in 300s for ~96% (vs airbench96's 37 epochs), so there is a large *epoch surplus* and the binding constraint is the **regularization ceiling**, not capacity or anneal length. The two prior capacity adds (EXP-005 deepen, EXP-007 widen) both lost by **under-annealing** — they cut epochs and the net finished still-climbing. The highest-EV direction is therefore a **throughput-free quality lever** that changes *what* the fully-annealed net converges to, without touching the step count.

Decoupled weight decay is exactly that. The causal chain:

1. **BN affine γ/β should not be L2-penalized.** Weight decay on a BatchNorm scale γ pulls it toward 0, which shrinks the effective magnitude of every post-BN activation. BN already controls activation scale by construction (it re-normalizes to unit variance then rescales by γ), so decaying γ is not a meaningful capacity-control regularizer — it is a *spurious* bias on the learned per-channel gain. "Bag of Tricks" §4 ("No bias decay") and the fastai/timm conventions exclude all 1-D params (BN γ/β and biases) from decay for precisely this reason: penalizing them "can lead to underfitting" and the benefit of L2 is concentrated on the high-dimensional weight matrices where it genuinely controls the function-class complexity.
2. **The penalty interacts with one-cycle's high peak LR.** At `PEAK_LR=0.4` the per-step decay increment `lr·wd·γ` is non-trivial during the ~85% of the budget spent above ~0.06 LR; γ is continuously dragged down and must be re-grown by the data gradient. Removing this frees γ/β to settle at their data-optimal values, which can sharpen the late-anneal optimum where most accuracy is set (EXP-001: "most accuracy lands in the low-LR tail").
3. **The ReZero `alpha` scalar is actively harmed by decay.** `GatedResidual.alpha` (shape [1], init 0) is the gate that ramps the layer2 ReZero block's capacity from identity. `weight_decay=5e-4` on `alpha` applies a constant restoring force toward 0 — i.e. it actively fights the block's capacity ramp that delivered EXP-004's +0.13pp. The data gradient on alpha is small (EXP-004 measured `α.grad≈0.0179`), so a `lr·wd·α` decay term is non-negligible relative to it. Excluding alpha lets the ReZero block reach a larger steady-state gate, plausibly recovering a bit more of the capacity EXP-004 added.

The net effect is a small reduction in *total* applied regularization (γ/β/α no longer decay) concentrated where decay was doing no useful work or active harm, leaving the decay on conv/fc weights — where it controls real complexity — intact. On a net that is regularization-bound but with surplus epochs, redistributing the penalty correctly is the kind of lever that can move a fully-annealed optimum.

## Concrete change (this codebase)

Single edit site: the optimizer construction in `main()`, `train.py:243-249`:

```python
optimizer = optim.SGD(
    [p for p in model.parameters() if p.requires_grad],  # exclude frozen whitening conv
    lr=PEAK_LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True,
)
```

Replace with a two-group partition by parameter dimensionality. I verified the parameter inventory against the actual model definition (`train.py:101-185`), not filenames:

- `whiten.weight` — 4-D, `requires_grad=False` (set at `train.py:147,169`). Must stay excluded; the partition filters on `p.requires_grad` first.
- All `conv_bn` Conv2d weights (`train.py:103`, `bias=False`) — 4-D → **decay group**.
- All `BatchNorm2d` (`train.py:104`) `weight` (γ) and `bias` (β) — both 1-D → **no-decay group**.
- `GatedResidual.alpha` (`train.py:134`, `torch.zeros(1)`) — shape [1], `ndim==1` → **no-decay group**.
- `fc` (`train.py:153`, `nn.Linear(512, 10, bias=False)`) — weight 2-D → **decay group**.
- There are **no bias parameters anywhere** in the net (every conv and the fc are `bias=False`), so the only 1-D params are BN γ/β and α. The `param.ndim <= 1` test is exact and robust here.

Proposed replacement:

```python
decay_params, no_decay_params = [], []
for p in model.parameters():
    if not p.requires_grad:          # skip frozen whitening conv
        continue
    if p.ndim <= 1:                  # BN γ/β and ReZero α scalar
        no_decay_params.append(p)
    else:                            # conv (4-D) and fc (2-D) weight matrices
        decay_params.append(p)

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

Notes for the planner:
- The LR-schedule loop at `train.py:290-291` already iterates `for g in optimizer.param_groups: g["lr"] = lr` — it writes LR to **every** group, so both groups get the identical one-cycle LR with no change needed. This is the only other place `param_groups` is touched.
- `momentum` and `nesterov` are passed as optimizer-level defaults and apply to both groups; `weight_decay` is overridden per-group. This is the standard PyTorch idiom.
- Keep `WEIGHT_DECAY = 5e-4` and `PEAK_LR = 0.4` as named constants unchanged (see tuning discussion below) so the diff stays a single-variable change.
- A 1-line smoke assertion is cheap and worth adding to the plan: `assert len(no_decay_params) > 0 and sum(p.numel() for p in no_decay_params) < 0.05 * learnable_params` — confirms the partition caught the BN/α params and that they are a small fraction of total (sanity that we didn't accidentally route conv weights into no-decay).

## On retuning PEAK_LR / WEIGHT_DECAY

**Recommendation: hold both fixed at 0.4 / 5e-4 for this experiment.** Rationale:

- **Single-variable attribution.** The whole value of this idea is a clean, throughput-free A/B against the EXP-004 baseline (96.00%). Co-tuning LR or wd would confound the attribution and forfeit that cleanliness.
- **Direction of the effective-wd change is favorable-leaning but small.** Decoupling *reduces* total applied regularization (γ/β/α stop decaying). The decay group still carries the full 5e-4 on the conv/fc weights — which is where the published recipes (johanwind, DavidNet) actually tuned 5e-4 to live. So the conv/fc regularization that sets the bulk of generalization is **unchanged**; only the spurious γ/β/α decay is removed. The standard result (Bag of Tricks Table 5; fastai/timm defaults) is that this *helps or is neutral* on already-regularized nets — it does not typically require re-raising wd to compensate, because the removed penalty was not doing useful regularization work.
- **Label-smoothing 0.2 is already heavy**, so the net is not starved of regularization; removing the γ/β/α decay is unlikely to tip it into overfitting in ~142 epochs. If anything, the risk is that the change is *too small to see* (below), not that it under-regularizes.
- A wd/LR sweep is the natural **follow-up** if this lands as neutral-to-slightly-positive: with γ/β decoupled, the conv/fc-only wd can often be raised (e.g. 5e-4→8e-4) for a further bump. But that belongs to a separate experiment, not this attribution test.

## Expected magnitude vs the ~0.1pp noise floor — honest assessment

This is the central risk and I will not inflate it.

- **Published effect size.** Bag of Tricks reports "No bias decay" as part of a bundle of tricks; the *isolated* contribution of no-decay-on-BN/bias in the literature is typically **~0.1–0.3pp** on ImageNet-scale nets, and is one of the more reliable of the "free" tricks. On small CIFAR ResNets the reported isolated effect is usually in the **+0.05 to +0.2pp** band.
- **This net is already heavily regularized** (label smoothing 0.2, Cutout 8, EMA, wd 5e-4) and **saturated near 96%**. On a saturated net, the *marginal* value of any single regularization correction shrinks — the easy generalization gains are already captured. That pushes the expected effect toward the **low end** (~0.05–0.15pp).
- **The α-decoupling is a second, partially-independent contributor** specific to this net: removing the restoring force on the ReZero gate is not in the generic Bag-of-Tricks accounting, and could add a few hundredths by letting EXP-004's block ramp further. This is the part of the idea that is *not* already priced into the saturated-net pessimism.
- **Verdict on magnitude:** plausibly a **>0.1pp** effect, but **not confidently so**. It sits in the contested zone right at the ~0.1pp noise floor. The honest framing: this is a *higher-probability-of-small-positive, lower-probability-of-clearly-clearing-the-bar* lever. Its real advantage over the capacity ideas is **asymmetry of downside** — it costs zero epochs, so unlike EXP-005/007 it cannot *lose* by under-annealing; the worst realistic case is "indistinguishable from baseline," not a regression.

## Throughput impact: zero (the key advantage)

Param grouping changes nothing in the forward/backward path or the per-step compute — SGD applies the same update math, just with `weight_decay=0` for two small param groups. Expected `num_epochs` is **unchanged at ~142–150** (subject only to the usual shared-host jitter). This sidesteps the under-annealing trap that sank EXP-005 (142→131) and EXP-007 (150→94) entirely. `num_epochs` is the first-class falsification diagnostic: if it comes in at ~142–150, throughput was preserved and any accuracy delta is attributable to the decoupling, not to an epoch-count confound.

## Verification / falsification

- **Attribution:** the only code change is the optimizer partition; training is otherwise byte-identical to EXP-004 (same seed 42, same schedule, same augmentation, same EMA/TTA). Confirm `num_epochs ∈ ~[135,155]` (throughput preserved) — this rules out the epoch-count confound that would otherwise muddy a capacity experiment.
- **Smoke check (off the official run):** print `len(decay_params)`, `len(no_decay_params)`, and `sum(numel)` per group; assert the no-decay group is BN γ/β + α only (small param count) and the decay group holds all conv/fc weight. One forward+backward step to confirm the optimizer constructs and steps without error.
- **Success:** `best_test_acc ≥ 96.10%` AND clearly above the ~0.1pp noise floor, with `num_epochs` in the normal band (so the gain is the decoupling, not a lucky high-epoch draw).
- **Falsification / null:** `best_test_acc ∈ [95.90, 96.05]` with normal epochs → effect is sub-noise (the saturated-net pessimism wins); record as no-improvement and conclude the BN/α decoupling does not clear the bar on this already-regularized net. A *regression* below ~95.85 with normal epochs would be surprising and would suggest the γ/β/α decay was load-bearing regularization (unlikely given LS 0.2) — worth noting if seen.
- **Confound guard:** because the metric sits near the noise floor, a single-run +0.07pp "win" is NOT a pass (per the established noise-floor protocol). Only a clear ≥0.1pp lift with normal epoch count counts.

## Strongest risk

**The effect is sub-noise on this saturated, already-heavily-regularized net.** The assumption that most needs to hold is that the γ/β/α decay was doing *net-negative or net-zero* work AND that removing it shifts the fully-annealed optimum by **more than ~0.1pp**. On a net already at 96% with LS 0.2 + Cutout + EMA, the marginal regularization correction may be worth only a few hundredths — landing in the null band and reading as no-improvement against a ~0.1pp bar that itself sits at the noise floor. The idea is *theoretically sound and standard practice* (so unlikely to hurt), but its magnitude on this specific saturated operating point is genuinely uncertain and may not clear the bar in a single seed-fixed run.

## Effort

**Low.** A single ~8-line edit at one site (`train.py:243-249`), no new dependencies, no schedule/architecture change, zero throughput cost, one 300s run plus a smoke check. Lowest-effort, lowest-downside-risk lever in the current candidate set; its weakness is upside uncertainty, not implementation cost or epoch risk.

## Evidence pointers
- **Bag of Tricks for Image Classification with CNNs**, He et al., CVPR 2019, arXiv:1812.01187 — §4 "No bias decay": exclude BN γ/β and biases from weight decay; cited in this goal's reference `knowledge/references/fast-cifar10-recipes.md:27` and flagged as a recipe lever ("BN/no-decay-on-bias") in `experiments/001/04-analysis.md:37` worth "tenths of a pp", never yet executed.
- **fastai / timm conventions:** both libraries default to no-decay on all 1-D params (the `no_weight_decay` / `bn_bias_no_wd` idiom) — the de-facto standard this proposal implements.
- **EXP-004 analysis** (`experiments/004/04-analysis.md`): ReZero α delivered +0.13pp with measured `α.grad≈0.0179≠0`; establishes that α is a live, accuracy-bearing parameter whose decay term is non-negligible relative to its gradient — the basis for the α-decoupling sub-mechanism.
- **EXP-005 / EXP-007 analyses** and the "under-anneal" Failed-Approaches entry (`03-experiment-learnings.md`): both capacity adds lost by cutting epochs; motivates the throughput-free, zero-epoch-cost design of this idea.
- **EXP-001 learning** (`03-experiment-learnings.md`, "most accuracy gain arrives in the low-LR tail"): supports the mechanism that a correctly-redistributed penalty acts on the fully-annealed optimum where 96.0 is set.
- **Noise-floor protocol** (`03-experiment-learnings.md` High-Importance): the ~0.1pp floor and the "treat sub-0.1pp single-run wins as unproven" rule, applied directly in the falsification criteria above.

## Out-of-scope confirmation
Edits only `train.py`; `prepare.py` untouched; no new dependencies (torch only); seed `torch.manual_seed(42)`/`torch.cuda.manual_seed(42)` unchanged; ≤1 eval/epoch unchanged; runs on GPU 1 (`CUDA_VISIBLE_DEVICES=1`).
