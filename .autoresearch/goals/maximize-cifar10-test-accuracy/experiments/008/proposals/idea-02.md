# Proposal (idea-02): Stronger data augmentation to spend the epoch surplus on a regularization-bound net

## One-line

Strengthen the train-time augmentation from `Cutout(8)` to **`Cutout(12)` + a light `transforms.RandomErasing(p=0.25, value=0.0)`**, leaving everything else byte-identical. This raises the regularization the saturated net fights against without touching GPU throughput, converting the diagnosed wasted-epoch surplus (~142–150 epochs vs airbench96's 37 for the same ~96%) into a higher annealed accuracy ceiling.

## Mechanism — how this advances `best_test_acc`, tied to the named limiter

The diagnosis names the limiter as **saturated / regularization-bound with a large epoch surplus**: the net fits ~142–150 epochs in 300s but airbench96 reaches ~96.03% in only 37 epochs, so the extra ~100 epochs are not buying accuracy — they are being spent re-fitting an already-fit training set (overfitting territory). The classic conversion of surplus epochs into accuracy on a saturated net is **stronger regularization/augmentation**: it makes each epoch's training distribution harder, so the net needs more epochs to reach the same training loss, and the fully-annealed minimum it lands in generalizes better. We have the epochs to pay for this.

The causal chain to the metric:
1. Cutout12 + RandomErasing erase more/larger regions of each image → the model can no longer rely on small localized cues → it is pushed to use more distributed, generalizing features (the documented cutout/random-erasing mechanism).
2. This slows *convergence* (more epochs to fit), which is exactly the wasted resource the diagnosis identifies. Because we still fully anneal the one-cycle LR to ~0 within 300s (the schedule is time-keyed, not step-keyed — `train.py:285-289`), the net reaches the flat, fully-annealed regime where most CIFAR accuracy is set (EXP-001 learning), but at a *lower-overfitting* minimum.
3. Net effect: the annealed `best_test_acc` ceiling rises.

The decisive structural reason this is safer than the recent capacity experiments (EXP-005, EXP-007 both failed by **under-annealing**): those changes slowed the **GPU step** (more FLOPs/step → fewer steps fit in 300s → truncated low-LR tail). **Augmentation is computed on the CPU `DataLoader` workers** (`NUM_WORKERS=8`, `persistent_workers=True`, `prefetch_factor=4`, `prepare.py:6`, `train.py:218-227`), in parallel with the GPU. Cutout and RandomErasing are cheap tensor slice-fills. So this change does **not** reduce img/s and does **not** cut the epoch count — it slows *convergence* (good, uses the surplus) without slowing *throughput* (avoids the EXP-005/007 trap). That is the core argument for why this lever is correctly matched to the diagnosis.

## Concrete change in THIS codebase

All edits are in `train.py`, in the `train_tf` pipeline built in `main()` (`train.py:205-213`), plus possibly one import line. Nothing else changes — architecture, optimizer, schedule, EMA, whitening, TTA, seeds, normalization stats all held byte-identical for clean single-variable attribution.

**Edit 1 — Cutout size 8 → 12** (the existing custom `Cutout` class, `train.py:42-61`, is already correct: it zeroes a square in normalized space, which equals the dataset mean in raw pixel space because `std=(1,1,1)`). Change only the constructor argument at the augmentation site:

```python
# train.py:211  (inside transforms.Compose)
Cutout(12),          # was Cutout(8)
```

This is a one-token edit and needs no class change. Note the class uses `s = self.size // 2` and clips with `max/min` (`train.py:57-60`), so `size=12` zeroes a 12×12 patch (clipped at borders) centered on a uniformly-random pixel — exactly the airbench96 `cutout=12` semantics.

**Edit 2 — add a light `RandomErasing` after `Cutout`** (torchvision built-in, already imported via `from torchvision import ... transforms`, so **no new dependency**). It operates on the post-`ToTensor`/post-`Normalize` CHW tensor, same space as `Cutout`. Critically set `value=0.0` so the erased region equals the mean in raw-pixel space (matching the existing Cutout-with-mean convention and the frozen mean-subtract/std=1 normalization):

```python
# train.py:205-213  -> train_tf becomes:
train_tf = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(EVAL_MEAN, EVAL_STD),
        Cutout(12),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15),
                                 ratio=(0.3, 3.3), value=0.0),
    ]
)
```

`RandomErasing` here adds, with probability 0.25, a second variable-size variable-aspect erased rectangle (area 2–15% of the image, capped well below the default 0.33 to stay controlled). On 75% of images it is a no-op, so the *combined* erasing is "always one 12×12 cutout, plus occasionally a second smaller erase" — a modest, well-targeted step up, not a heavy overhaul. `RandomErasing` draws from torch's global RNG; the fixed seed (`train.py:199-200`) is untouched, so this is not seed hacking.

Nothing in the forward pass, eval path (`evaluator.evaluate`, `prepare.py`), or schedule changes; `num_params` is identical; `prepare.py` is byte-unchanged.

## Evidence

- **Cutout12 is the documented airbench96 value** at ~96.03% (EXP-007 brainstorm §Web Search, citing `airbench96.py`: "Augmentation: flip + translate=4 + **cutout=12**"; `knowledge/references/fast-cifar10-recipes.md`). We currently run cutout 8 (`train.py:211`). This is a direct, evidence-backed step toward the reference recipe along the *augmentation* axis — and the only such axis the reference uses that we under-shoot (we already match label_smoothing 0.2 and crop pad 4).
- **The epoch surplus is real and large.** EXP-004/006 fit 142/150 epochs in 300s (`03-experiment-learnings.md` Protocol Findings); airbench96 reaches 96.03% in 37 epochs. The surplus (~4×) is the resource stronger augmentation is designed to consume. Web search confirms heavy augmentation "benefits from longer training schedules and is less prone to overfitting" — i.e. it is the matched tool for a many-epoch saturated regime.
- **Augmentation is throughput-free here, unlike the failed capacity adds.** EXP-005 (−0.10pp) and EXP-007 (−0.15pp) both failed by *under-annealing* because they slowed the GPU step (`03-experiment-learnings.md` Failed Approaches, Medium: "any capacity add cuts the step count"). This change adds CPU work on 8 parallel workers, not GPU work, so it does not cut steps. The diagnosis explicitly flags this property ("throughput-cheap because augmentation runs on CPU DataLoader workers ... in parallel with the GPU").
- **RandomErasing is a peer-reviewed, label-preserving regularizer** (Zhong et al., "Random Erasing Data Augmentation", AAAI 2020) and is a first-class torchvision transform, so it adds no dependency. With `value=0.0` it is mean-fill in this normalized space, identical in spirit to the existing `Cutout`, just probabilistic and variable-size — a controlled increment rather than a new mechanism.

## Strongest risk

**The combined effect is sub-noise (the EXP-006 fate), not under-annealing.** The ~0.1pp run-to-run noise floor (`03-experiment-learnings.md` Protocol Findings, HIGH) is the dominant threat. Two specific concerns:

1. **Cutout12 alone is likely sub-noise.** Going 8→12 is a small recipe-alignment tweak; airbench's 96.03 vs our 96.00 is only +0.03pp and confounded by width/GELU/3-conv-groups (EXP-007 review point #2). A single-run A/B on cutout12-alone would very plausibly disappear under epoch-count jitter, exactly like EXP-006's real-but-sub-noise TTA gain. **This is why the proposal combines cutout12 with RandomErasing** — to stack two regularizers so the summed effect has a better chance of clearing >0.1pp. I am honestly flagging that *either tweak alone is a weak single-run bet*; the combination is the bet.
2. **Over-regularization / under-fit.** If the combined erasing is too strong for ~142 epochs, the net could finish slightly *under-fit* (train loss not low enough) and the gain cancels or goes negative. This is mitigated by keeping RandomErasing light (p=0.25, area ≤15%) and is *observable*: watch whether the early/mid trajectory falls noticeably below the EXP-004 curve (e.g. ep25 should still be near ~92%) and whether `best == final` with a still-rising tail (the under-anneal signature). Because throughput is preserved, true under-*annealing* (truncated tail) should NOT occur — only mild under-*fitting* if aug is excessive, which the conservative settings guard against.

**The assumption that most needs to hold:** that the net is genuinely *overfitting* in its surplus epochs (so harder augmentation helps) rather than already optimally regularized at cutout8 (in which case more erasing only hurts). The diagnosis asserts the former; this experiment is its direct test.

## Why NOT the heavier options (honest evaluation)

- **(c) TrivialAugmentWide / RandAugment / AutoAugment** (torchvision, on PIL uint8 *before* ToTensor): strongest regularizers, but designed for *long* schedules (100s of epochs in their original papers; web search: they "benefit from longer training schedules"). At ~142 epochs with a one-cycle that fully anneals only once, they risk pushing the net into under-*fitting* (the EXP-007-style "still climbing at budget end", but caused by aug strength rather than lost throughput). They also change the augmentation distribution drastically (geometric + color ops), a multi-variable jump that conflicts with clean attribution under the noise floor. **Verdict: too aggressive for a 300s single-anneal budget; higher variance than the modest combination.** A reasonable *future* probe if cutout12+erasing clearly under-regularizes (net still overfitting), but not the controlled first step.
- **(d) RandomCrop padding 4 → more**: pad 4 is the well-established CIFAR optimum (airbench, DavidNet, hlb all use translate/pad 4); larger pad mostly shifts the object off-frame and tends to hurt. Sub-noise at best. **Skip.**
- **(e) mixup / CutMix** (loss/target mixing, implemented in the training loop, not the transform): genuinely interesting for a saturated net, but high implementation risk *here* — it interacts with `label_smoothing=0.2` (`train.py:250`, double soft-targeting), the weight EMA (`AveragedModel`, `train.py:254-256`), the time-based schedule, and `Eval.evaluate`'s hard `cross_entropy` (`prepare.py:43`). torchvision's `v2.MixUp`/`CutMix` need a collate-function change and v2 transforms. This is a multi-variable change requiring care and is a larger effort; the fast-CIFAR lineage deliberately avoids mixup in short budgets (EXP-007 brainstorm: "short-budget fast-training avoids it"). **Defer** — out of scope for a controlled single change this loop.

## Throughput cost estimate

Near-zero GPU impact. Current steps run at ~26k img/s (~20ms/step at batch 512; EXP-004 log), i.e. **GPU-bound** with bf16+channels_last. Cutout is already in the pipeline (`train.py:211`); changing 8→12 is the same slice-fill at a slightly larger slice. RandomErasing on 25% of images adds one extra variable-size slice-fill on the CPU worker. With 8 persistent workers and prefetch_factor=4 staying ahead of a 20ms GPU step, the augmentation latency is hidden. **Expectation: img/s and num_epochs effectively unchanged vs EXP-004 (~142–150), modulo the shared-host jitter that already causes the ±0.1pp floor.** This is the key throughput claim and is directly verifiable from the run's `img/s` and `num_epochs` lines.

## Expected magnitude vs noise floor

- **Direction:** positive if the overfitting premise holds; the regularization mechanism is well established.
- **Magnitude:** modest. Cutout8→12 alone ≈ a few hundredths pp (likely sub-noise). Adding light RandomErasing is intended to push the *combined* effect to roughly +0.1–0.2pp — i.e. *at or just above* the bar, not comfortably clear of it. **I am being candid: this is a marginal-headroom bet.** Its appeal over the capacity ideas is that it carries essentially *zero downside-from-under-annealing risk* (throughput preserved), whereas EXP-005/007 each lost 0.10–0.15pp purely to lost epochs. So the realistic outcome distribution is "small win or small/zero loss", a better risk profile than the capacity bets even if the upside ceiling is lower.

## Verification / falsification

- **Pre-register `num_epochs` and `img/s` as first-class evidence** (the EXP-007 protocol). The *positive* check that the mechanism is the intended one: epochs stay ~142–150 (throughput preserved) AND the early/mid trajectory is at or modestly below EXP-004's (slower convergence from harder aug), with the tail closing the gap and finishing higher. If epochs *drop* substantially, something other than CPU aug changed and the run is confounded.
- **Improvement (C3):** `best_test_acc ≥ 96.10` and clearly above the ~0.1pp floor. Cross-check `best == max per-epoch eval == summary`, one eval/epoch, seeds unchanged, `prepare.py` byte-unchanged, diff confined to the two augmentation lines.
- **Falsifiers:**
  - best < 96.10 with epochs ~142–150 and a *fully-annealed* tail (best != final, or best==final but trajectory flat at the end) → augmentation is sub-noise / the net was already optimally regularized at cutout8 → the overfitting premise is wrong, pivot to capacity.
  - best < 96.10 with `best == final` and a *still-rising* tail → under-*fitting* from too-strong aug (not under-annealing, since epochs held) → back off RandomErasing (drop it / lower p) or cutout to 10.
  - epochs collapse well below ~140 → throughput *was* affected (unexpected); the CPU-free premise failed → investigate worker saturation before re-judging.

## num_epochs expectation

~142–150 epochs (unchanged from EXP-004/006), because the change is CPU-side and does not slow the GPU step. This is the central prediction and the cleanest single diagnostic distinguishing this lever from the EXP-005/007 under-anneal failures (which dropped to 131 / 94).

## Effort

**Low.** Two lines in `train.py` (one Cutout argument, one added `RandomErasing` transform; `RandomErasing` is already importable via the existing `transforms` import). One run under `timeout 600 CUDA_VISIBLE_DEVICES=1`. No architecture, schedule, loop, or eval changes; fully byte-reversible; zero baseline risk from the change itself.
