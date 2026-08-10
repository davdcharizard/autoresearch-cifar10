# Proposal: Fixed GeM-3 Final Pooling

## Claim and exact intervention

Test one parameter-free readout change on the accepted EXP-010 recipe: replace global average pooling of the final nonnegative post-ReLU `128x8x8` map with fixed generalized-mean pooling at `p=3`. For final activations `x >= 0`, define each pooled channel as

`g[n,c] = ((1/64) * sum_{h=1..8,w=1..8} max(x[n,c,h,w], 1e-6)^3)^(1/3)`.

The production expression is exactly:

```python
GEM_P = 3.0
GEM_EPS = 1e-6

# after the existing final block and its ReLU
out = out.clamp_min(GEM_EPS).pow(GEM_P).mean(dim=(-2, -1)).pow(1.0 / GEM_P)
return self.fc(out)
```

Compute entirely in the existing FP32 path. Do not learn or tune `p`, subtract epsilon after the root, add a max/average branch, normalize features, or rescale logits. Keep the same `Linear(128,10)` object, initialization order, bias, 1,073,962 total parameters, optimizer parameter ordering/state, and all model/data RNG consumption. GeM creates no module, parameter, buffer, or random draw; reset-seed control and candidate models must therefore have bitwise-identical parameters/buffers and identical CPU/CUDA RNG states after construction.

Preserve everything else from EXP-010: width-2 postactivation ResNet-20, Option-A shortcuts, batch 128, seed 42, N1/M7 plus alpha-1 CutMix on 50% of strong batches through the 80% boundary, hard weak tail, SGD `lr=0.1`, momentum `0.9`, all-parameter decay `1e-4`, LR schedule, timer, worker lifecycle, and evaluation cadence.

## Mechanism and hypothesis

Average pooling weights all 64 positions equally and can dilute compact class-bearing responses. GeM-3 is a smooth intermediate between average (`p=1`) and max (`p -> infinity`): its activation gradient is distributed across positive positions in proportion to squared magnitude, so salient regions receive more weight without the raw max path's single-index gradient or independent classifier. Patch extent still affects the power mean, making it less area-insensitive than max under CutMix.

The hypothesis is that this fixed salience bias preserves EXP-010's strong-phase health and at least 97% of its 26,898 updates while raising `best_test_acc` from 94.15% to at least 94.25%. Evidence is indirect: GeM improved global visual descriptors, while the local mixed-pooling review supports adaptive spatial statistics but not this exponent for CIFAR classification (`knowledge/papers/mixed-pooling.md`; EXP-014 `01-brainstorm.md`).

## Scale and first-update safety gates

Use disposable diagnostics on an immutable corpus of at least 16 real production-distribution batches, containing both hard and alpha-1 CutMix targets. Load identical accepted weights into control and candidate and preserve the same optimizer state. No diagnostic hook may remain in production.

Before any update, record per-example pooled-feature L2 norms, logit RMS, loss, class histogram, and classifier-gradient norm. Authorize timing only if all values are finite and, on every batch:

- median and p95 ratios `||g_GeM||2 / ||g_avg||2` are at most `2.0` and `2.5`, respectively;
- candidate/control logit RMS and `||grad(W_fc)||2` ratios are each at most `2.0`;
- candidate loss is at most `1.5x` control and neither arm assigns more than 95% of predictions to one class.

Then run paired one-step probes from independently restored identical model/SGD states on every batch. Measure each arm's RMS logit displacement on the same batch after its standard `lr=0.1` update. Require candidate/control displacement at most `2.0`, candidate post/pre loss at most `2.0`, finite parameters/BN buffers/momentum, and no candidate-only class concentration above 95%. Continue one aligned arm per method for 200 corpus steps and require no nonfinite state, candidate loss EMA at most `1.5x` control, and no candidate-only >95% concentration. These gates directly reject the EXP-014 mechanism: its independent raw-max classifier had a `4.10x` first gradient, a `10x` same-batch loss jump, and one-class collapse after one update. Passing does not predict accuracy; it only establishes bounded optimization continuity.

## Paired timing and production verification

On one idle 97,871-MiB H20, use persisted post-transform batches so both arms see byte-identical strong hard/CutMix and weak hard inputs. Run five fresh-process pairs with AB/BA order alternation. Each arm restores the same model/optimizer state, warms up for 100 steps, then measures at least 1,000 complete synchronized steps including H2D, forward, loss, backward, SGD, and final synchronization. Measure strong and weak paths separately and combine means 80/20.

Proceed only if the weighted candidate/control mean is at most `1.025`, every pair is below `1.04`, per-arm trial-mean CV is at most 2%, candidate p95 is at most `1.05x` control, projected exposure is at least 26,242 steps, peak allocation is below 650 MiB, and the conservative 19-evaluation wall projection is below 540 seconds. Separately confirm finite evaluator outputs and at most `1.10x` accepted inference time. GeM's cube, reduction, root, and backward are counted costs; kernel fusion or algebraic substitution is not allowed.

If all gates pass, run seed 42 exactly once with output only in `run.log`; no retry or exponent/epsilon adjustment. Require exit zero, 300 counted seconds, total below 600 seconds, at least 26,242 steps, 1,073,962 parameters, one 80% switch with eight workers stopped, expected target formats, and unique at-most-once-per-epoch evaluations. Record the switch checkpoint; below 87.08% diagnoses strong underfit but cannot stop, tune, or rerun the experiment.

## Risks and falsification

GeM can amplify RandAugment artifacts, overweight small CutMix donor regions relative to area labels, raise feature/logit scale, starve low activations through the epsilon clamp, or spend enough backward time to reduce useful exposure. Retrieval evidence may not transfer to balanced small-image classification, and fixed `p=3` may simply impose the wrong spatial prior.

Any semantic, scale, first-update, concentration, timing, memory, or wall gate failure falsifies this exact operating point before production. If production exposure falls below 26,242, the efficiency hypothesis is false even if accuracy rises. If integrity and exposure pass but `best_test_acc <94.25%`, record valid no-improvement and retire fixed `p=3`, `epsilon=1e-6` GeM on this recipe. Only `best_test_acc >=94.25%` with every gate passed supports acceptance; a bare 0.10-point single-seed gain remains weak causal evidence.
