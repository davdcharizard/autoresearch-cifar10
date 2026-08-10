# Proposal: Shared-Budget CutMix and Manifold Mixup

## Summary

Preserve EXP-004's independent 256-image batches, total early mixing probability, clean final quarter, and period-two SAM tail. During the first 75% of charged time, keep one `p=0.5` mixing gate. Allocate 75% of selected batches to the parent's CutMix and 25% to one-pass manifold mixup at either of the first two WRN stage boundaries. This preserves most validated CutMix exposure while introducing a representation-level mechanism without adding a forward pass, changing the clean/mixed ratio, or repeating EXP-005's identity-throughput failure.

## Exact Policy

For every batch with `progress < 0.75`:

1. Draw one policy gate. With probability `0.5`, leave the batch clean.
2. If selected, draw one mode. With probability `0.75`, apply the existing CutMix helper unchanged. Otherwise apply manifold mixup.
3. For manifold mixup, choose uniformly between the hidden boundaries after block 2 and block 4 in one-based block count: after the complete 64-channel stage (`blocks[0:2]`) or after the complete 128-channel stage (`blocks[0:4]`).

The marginal early probabilities are therefore:

```text
clean                         0.500
CutMix                        0.3750
manifold after 64ch stage     0.0625
manifold after 128ch stage    0.0625
```

This is one fixed policy, not a sweep. It retains the parent's overall 50% early mixed-batch budget and 50% clean supervision while preserving 75% of its CutMix exposure. At `progress >= 0.75`, all four mixing probabilities become zero and the exact EXP-004 clean-tail SAM policy runs.

## Manifold Mixup Math

For a selected manifold batch, sample one permutation `pi` and one scalar

```text
lambda ~ Beta(2, 2)
```

for the whole minibatch. At the selected hidden boundary, after both independently augmented images have passed through all preceding layers, mix

```text
h_mix = lambda * h + (1 - lambda) * h[pi]
```

and continue the remaining layers once. The paired targets are `y_b = y[pi]`, and the single-logit loss is

```text
loss = lambda * CE(logits, y) + (1 - lambda) * CE(logits, y_b)
```

Unlike CutMix, manifold lambda is not area-corrected: the representation interpolation is exactly linear, so the sampled coefficient is the true label coefficient. Use `alpha=2`, matching the main CIFAR setting in `experiments/006/papers/manifold-mixup.md`. Both source representations participate in the same batch computation and gradient graph, so the operation preserves all images and adds neither examples nor model forwards.

To sample `Beta(2,2)` with an explicit generator and no dependency, use the integer-shape gamma identity. Draw four independent uniforms from a private CPU generator, clamp only away from zero for `log`, then compute

```text
g_a = -log(u1) - log(u2)   # Gamma(2, 1)
g_b = -log(u3) - log(u4)   # Gamma(2, 1)
lambda = g_a / (g_a + g_b)
```

This is an exact Beta(2,2) construction, not a uniform approximation. Return a Python float in `(0,1)` for the activation and loss coefficients.

## Model-Forward API

Modify only `train.py`. Extend `PreActWideResNet.forward` with optional arguments whose defaults preserve evaluator compatibility:

```python
def forward(
    self,
    x,
    drop_scale=0.0,
    mix_boundary=None,
    mix_permutation=None,
    mix_lambda=1.0,
):
```

`Eval.evaluate(model, device)` and all clean/CutMix/SAM calls continue to use the defaults, so inference is unchanged. In the block loop, count completed blocks from one through six. Immediately after completed block 2 or 4, if it equals `mix_boundary`, perform one out-of-place mix:

```python
paired = out[mix_permutation]
out = mix_lambda * out + (1.0 - mix_lambda) * paired
```

Out-of-place construction avoids corrupting the paired source. Assert that all three manifold arguments are supplied together, that the boundary is in `(2,4)`, that the permutation is on the same device with batch shape, and that exactly one boundary was applied. Do not expose stem, within-stage, final-stage, or classifier mixing in this experiment. These two boundaries map the paper's first two hidden choices onto the natural stage structure of this six-block WRN.

The training loop sets `targets_b` and a common `mix_lambda` for either mixed mode. CutMix still mutates inputs and returns its clipped-area `adjusted_lam`; its model call uses no hidden-mix arguments. Manifold mixup leaves inputs untouched and passes its boundary, permutation, and sampled lambda to the one model call. The existing two-term loss then uses the mode-specific coefficient.

## RNG Contract

Keep all stochastic policy draws off the global RNG used by shuffle, crop/flip, and drop path.

- Retain the parent's dedicated seed-42 CPU/CUDA CutMix generators for CutMix lambda, center, and permutation.
- Add a CPU policy generator with fixed seed `CUTMIX_SEED + 1` for the early gate, selected mode, and hidden-boundary choice.
- Add dedicated CPU and CUDA manifold generators with fixed seed `CUTMIX_SEED + 2` for the four-uniform Beta draw and permutation.

The offsets are fixed namespaces, not searched seeds. A clean batch consumes only its policy-gate draw. A CutMix batch consumes a policy mode draw plus the existing CutMix geometry/permutation draws. A manifold batch consumes a policy mode and boundary draw plus exactly four manifold CPU uniforms and one manifold CUDA permutation. No branch consumes another branch's generator. The model still executes every block once, so drop-path draw count is unchanged even when mixing occurs at a hidden boundary.

## Parent Compatibility and SAM

Keep `CUTMIX_PROB=0.5`, `CUTMIX_ALPHA=1.0`, `CUTMIX_END=0.75`, WRN architecture, optimizer, LR, drop-path schedule, data loader, seed 42, and evaluation cadence. Introduce `MANIFOLD_SHARE=0.25`, `MANIFOLD_ALPHA=2.0`, `MANIFOLD_BOUNDARIES=(2,4)`, and explicit generator seeds in the config log.

Manifold and CutMix are both forbidden at `progress >= SAM_START=0.75`. Preserve and generalize the existing assertion so `apply_sam` cannot coexist with either mixed target. SAM's first and perturbed calls use the default forward API, hard labels, CUDA RNG replay, disabled second-pass BatchNorm tracking, exact parameter restoration, and one optimizer update. There is no mixed hidden state to replay during SAM because the phases do not overlap.

All activation mixing occurs inside the existing charged `t0`/CUDA-synchronization interval. The largest selected boundary has a paired activation gather of roughly 32 MiB in BF16 after the 64-channel stage; it occurs on only 6.25% of early batches, with the later boundary half that size. There is no second forward, sampler change, or extra image transform. Expected overhead is below 2%, with optimizer exposure near EXP-004's 25,560 steps and comfortably above a preregistered 24,000-step floor.

## Evidence and Causal Rationale

The parent is generalization-limited and already preserves independent-image throughput. Manifold mixup encourages flatter class-conditional hidden representations and smoother decision boundaries; the paper reports CIFAR-10 error reductions from 4.83 to 2.95 on PreActResNet-18 and 3.99 to 2.55 on WRN-28-10, with no extra forward (`experiments/006/papers/manifold-mixup.md`). Its strongest eligible set includes linear input Mixup plus the first two hidden boundaries. This experiment deliberately uses CutMix rather than linear Mixup for the input-space component, so it is a hybrid adaptation and does not reproduce the paper's `{0,1,2}` policy.

EXP-005 fell to 95.28% despite retaining nearly the same number of steps because its overlapping sampler halved new-identity introduction (`experiments/005/04-analysis.md`). This proposal leaves the parent's DataLoader and 50,000-image epoch semantics unchanged. EXP-003 also showed 0.14-0.29-point selected-run variation, so the fixed policy must be run once without selecting among shares, alphas, or boundaries.

Attribution is intentionally limited: a result measures the combined policy of replacing one quarter of selected CutMix batches with equally distributed hidden manifold mixing. It cannot isolate manifold mixup from reduced CutMix exposure or identify which boundary contributed. Holding total mixed exposure, clean exposure, image throughput, and late SAM fixed makes that combined policy comparison interpretable; claiming a pure manifold effect would not be justified.

## Expected Effect and Hypothesis

EXP-004 is 95.40%, so success requires `best_test_acc >= 95.50%`. After heavily discounting the source's longer, weaker-baseline results, the hypothesis is that a 12.5% marginal manifold dose will complement retained CutMix and improve the solution entering SAM by 0.15-0.40 points, yielding 95.55-95.80% while retaining at least 24,000 optimizer steps. Final accuracy close to best and final loss no worse than the parent's 0.1654 are supportive only; they do not replace the primary gate.

## Strongest Risks

- Reducing validated CutMix exposure by 25% may cost more than manifold mixup adds; the experiment tests the replacement policy, not a free additive mechanism.
- The paper's large gains compare manifold mixup with weaker recipes and deeper networks. This already regularized WRN-16-4 may have less headroom, and only two early boundaries are available.
- `Beta(2,2)` concentrates around strong 50/50 mixtures. Combined with drop path and early CutMix, hidden targets may underfit despite the final clean quarter.
- Mixing after BatchNorm and stochastic residual blocks changes downstream activation statistics. This is intended, but its composition with late SAM is not directly evidenced.
- A 0.10-point threshold is smaller than EXP-003's observed selection variation. No retry, alternate share, alpha, boundary set, or seed may be chosen after the result.
- An API error could silently mix twice, use a CutMix area coefficient for hidden states, mutate the source activation, or activate during evaluation/SAM; focused assertions and smokes are required.

## Verification and Smokes

Before the full run:

1. **Policy frequencies**: simulate the private generators for a fixed large count and verify probabilities near 0.50 clean, 0.375 CutMix, and 0.0625 per hidden boundary; verify deterministic counter replay.
2. **Beta sampler**: compare fixed uniforms against the gamma formula, check finite `(0,1)` output, deterministic generator advancement by exactly four draws, and a large-sample mean near 0.5.
3. **Forward defaults**: in eval mode, verify `model(x)` equals `model(x, mix_boundary=None)` and the frozen evaluator call needs no change.
4. **Boundary exactness**: instrument fixed toy blocks or hooks to prove mixing occurs once after block 2 or 4, never at another location; invalid/incomplete argument combinations must raise.
5. **Representation/label math**: use source-coded hidden tensors and a cyclic permutation to verify `h_mix`, paired target orientation, two-term loss, and gradients to both unpermuted and permuted source rows.
6. **No aliasing**: verify manifold mixing does not mutate the pre-mix activation or input batch and fixed-point/cyclic permutations use pristine sources.
7. **CutMix regression**: rerun existing patch orientation, clipped-area lambda, zero-area, and generator tests; manifold batches must not advance CutMix generators and vice versa.
8. **Global RNG isolation**: snapshot global CPU/CUDA states around policy/lambda/permutation helpers and verify no change. In a training smoke, verify the model consumes the same count of global drop-path draws for clean and manifold calls.
9. **GPU integration**: run BF16/channels-last clean, CutMix, both hidden boundaries, and late SAM steps. Check finite loss/gradients, one forward per non-SAM step, no mix/SAM overlap, one BatchNorm update per primary pass, exact SAM restore, and one optimizer update.
10. **Audit/static checks**: compile and lint `train.py`; confirm only it changes; log eligible, clean, CutMix, boundary-2, and boundary-4 counts plus unchanged SAM counters and complete config.

Run exactly once after confirming physical GPU 0 is the approximately 98 GB NVIDIA H20:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

Verify exit 0, no NaN/Inf or CUDA errors, 300-second charged budget, total runtime below 600 seconds, one evaluation per completed epoch, `num_params=2,748,890`, `num_steps>=24,000`, early counters matching the preregistered partition, first SAM progress near 0.75, exact period-two SAM arithmetic, full summary, and `best_test_acc>=95.50%`. Remove `run.log` after analysis. Do not rerun or tune from test accuracy.

## Effort

**Medium.** The runtime change is localized and cheap, but safely extending the forward API, isolating RNG streams, preserving CutMix/SAM behavior, and proving exact boundary and label semantics require focused integration tests.
