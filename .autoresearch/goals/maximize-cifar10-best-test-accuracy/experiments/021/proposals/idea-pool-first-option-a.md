# Idea: Deterministic Pool-First Option-A Shortcuts

## Exact Candidate

Change only the two stride-2 Option-A shortcuts in the accepted width-2
postactivation ResNet-20. Replace raw even-phase slicing with fixed,
non-overlapping `2x2` average pooling, then retain the exact existing
high-channel zero pad:

```python
shortcut = x
if self.need_pad:
    shortcut = F.avg_pool2d(
        shortcut,
        kernel_size=2,
        stride=2,
        padding=0,
        ceil_mode=False,
        count_include_pad=False,
    )
    shortcut = F.pad(
        shortcut, (0, 0, 0, 0, 0, self.pad_channels)
    )
out += shortcut
return F.relu(out)
```

The current `shortcut[:, :, ::2, ::2]` line is removed. No module, learned
projection, normalization, coefficient, padding mode, or alternate pool is
added. The accepted residual branch remains byte-equivalent, including its
stride-2 `3x3` convolution. All seven same-shape shortcuts remain exact raw
identities.

Pin the transition inventory:

| Block | Shortcut input | Pool output | Channel pad | Shortcut output |
|---|---|---|---|---|
| `layer2[0]` | `32x32x32` | `32x16x16` | 32 zero channels | `64x16x16` |
| `layer3[0]` | `64x16x16` | `64x8x8` | 64 zero channels | `128x8x8` |

Every pool window is anchored at `(0,0)`, has no overlap or padding, and uses
exactly four input positions. Every spatial input contributes to exactly one
shortcut output. The first `C` output channels retain their original channel
identity as cell means; the added high `C` channels remain exactly zero.

## Why This Is a Clean Follow-Up

EXP-010 remains the 94.15% frontier with 26,898 updates, a 89.73% switch
checkpoint, 93.16% first weak checkpoint, and 0.1934 final NLL. Width 2 and
conservative CutMix are the validated representation/regularization mechanisms.

The accepted Option-A shortcut discards three of every four spatial positions
at both stage transitions. Pool-first Option A instead transmits the local mean,
which is a fixed low-pass statistic with all-position coverage. It may preserve
small translations and reduce phase sensitivity from crop/flip/RandAugment
without suppressing residual branches or changing channel provenance.

EXP-017 bundled the same `2x2` pool with random learned `1x1` channel mixing and
a new BN. It improved switch/first-weak fit by 0.47/0.29 points but worsened NLL
and peak accuracy. This candidate removes the two likely confounds: no random
shortcut basis and no strong-to-weak shortcut-BN state. It isolates deterministic
spatial reduction while preserving Option-A's fixed channel map. A success would
support pooling; a loss would reject this exact box-filter shortcut, not learned
ResNet-D or full-path anti-aliasing.

ResNet-D provides directional evidence that pooling before shortcut projection
can preserve transition information. Direct CIFAR-10 ResNet-20 research also
finds downsampling configuration accuracy-critical. Neither source estimates
this parameter-free Option-A variant, so the effect must be measured locally.

Sources:

- `experiments/017/papers/resnet-d-downsampling.md`
- `experiments/016/papers/resnet20-downsampling-search.md`
- EXP-017 `04-analysis.md`

## Architecture, State, and RNG Invariants

The candidate remains exactly **1,073,962 trainable parameters**. Convolution,
BN, and classifier modules; state-dict keys/shapes/values; optimizer membership;
and Kaiming initialization are unchanged. `F.avg_pool2d` owns no parameter or
buffer and consumes no CPU/CUDA random number. From cloned seed-42 state,
candidate/control construction must leave all learned state and post-construction
RNG bitwise identical. Data shuffle, worker seeds, augmentations, and CutMix
therefore begin from the accepted stream.

Preserve the complete EXP-010 recipe: batch 128; FP32; SGD momentum 0.9;
all-parameter decay `1e-4`; LR 0.1 through 80%, then 0.01 cosine to `1e-4`;
N1/M7 plus p=0.5 alpha-1 CutMix during the strong phase; hard weak tail; eight
persistent workers and explicit switch shutdown; fixed evaluator; seed 42;
300 counted seconds; at most one evaluation per epoch; and existing logging.

The fixed pools compute 12,288 outputs or 49,152 source contributions per image:

```text
32*16*16 + 64*8*8 = 12,288 pooled values
12,288 * 4 = 49,152 source contributions
```

This is negligible beside about 161.3M accepted forward MACs, but two pool
forward/backward kernels can still be launch-bound and must pass paired timing.

## Semantic and Safety Gates

Before any full run, require:

1. Exactly two `avg_pool2d` calls are reachable, only in stride-2/channel-double
   blocks. No pool is reachable from the stem, residual path, same-shape blocks,
   or final readout.
2. For seeded inputs, transition shapes match the table and residual/shortcut
   additions match exactly. Nontransition block outputs are bitwise equal to an
   aligned control.
3. Candidate shortcut output equals
   `F.pad(F.avg_pool2d(x, 2, 2), channel_pad)` exactly for both transitions.
4. A coordinate-ramp test proves each retained output is the arithmetic mean of
   its exact `2x2` cell. Four impulse tests within one cell each produce `0.25`
   at the same pooled coordinate; no adjacent output changes.
5. For the sum of retained shortcut outputs, autograd gives gradient `0.25` to
   all four cell positions. Padded channels remain zero and introduce no source
   gradient. The control slice instead gives weight one only to the even phase,
   proving the intended semantic difference.
6. Hooks prove both accepted stride-2 residual branches, every learned tensor,
   all BN counters, and hard/probability-target CE behavior remain finite and
   structurally unchanged.
7. Parameter count, state dictionary, optimizer groups, and post-construction
   CPU/CUDA RNG are bitwise aligned; syntax, Ruff, formatting, pre-commit, and
   tracked-scope checks pass.

For numerical safety, persist one exact 200-batch production N1/M7 hard/soft
sequence and replay it through aligned control/candidate training processes.
Require matching serialized batch hashes, finite state/loss, no candidate-only
class concentration above 95%, and candidate terminal loss EMA no more than
1.5x control. This is a collapse veto, not an accuracy proxy or pool selector.

## Timing and Exposure Vetoes

On the sole idle H20, run five alternating fresh-process control/candidate pairs
with identical weights, batch 128, alternating hard/probability targets, 100
warmups, and 500 measured synchronized production-region steps. Measure H2D,
forward, loss, backward, SGD, and synchronization. Separately measure inference.

Require:

- candidate/control median trial-mean training ratio `<=1.02`;
- `floor(26_898 * control_mean / candidate_mean) >=26_360`, retaining at least
  98% of accepted updates;
- candidate p95 `<=1.05x` control, trial CV below 3%, finite state, and peak
  allocation below 625 MiB and no more than 16 MiB above control;
- inference ratio `<=1.02`, projected total wall below 540 seconds, and projected
  evaluator count no more than EXP-010's 19.

Any semantic, safety, timing, memory, exposure, wall, or evaluator-count failure
is a no-launch veto. Do not remove pooling from one transition, change to
`3x3`, add reflection padding, pool the residual path, or recover cost with a
different precision. Those are new experiments, not fallbacks.

## Hypothesis, Risks, and Decision Rule

**Hypothesis:** deterministic `2x2` pool-first Option-A shortcuts will reduce
transition phase loss while preserving accepted channel identity, residual
activity, and at least 98% of optimizer exposure, raising `best_test_acc` from
94.15% to at least **94.25%**. A plausible success range is 94.25-94.40%.

Main risks:

- CIFAR details are small; box filtering can erase class-bearing edges and
  textures or dilute CutMix boundaries.
- Average pooling is not lossless: it retains all-position mean information but
  discards within-cell arrangement.
- The shortcut no longer transports selected activations exactly. Its backward
  gradient is spread at 0.25, weakening direct identity magnitude at transitions.
- Only the shortcut is filtered; the residual stride-2 convolution remains
  phase-sensitive, so branch spectra can be mismatched at addition.
- Crop padding and zero-valued borders can lower pooled boundary magnitude.
- Direct literature evidence does not isolate this shallow Option-A variant;
  the expected effect is near the one-seed resolution floor.

Mechanism diagnostics are switch accuracy versus 89.73% and the 87.08% underfit
marker, first weak versus 93.16%, final NLL versus 0.1934, step retention, and
best-final slope. They never override the primary metric.

After all vetoes pass, run once under the goal's H20/600-second protocol with
output only in `run.log`. Require exit zero, finite summary, approximately 300
counted seconds, total below 600, exact parameter count, one switch with eight
stopped workers, 45-55% CutMix, hard weak targets, unique epochs, at most one
evaluation per epoch and at most 19 total, seed 42, and only reviewed `train.py`
changes. Accept only `best_test_acc >=94.25%`; a correct lower run is valid
no-improvement. Never reroll, tune, or run a fallback variant.
