# Idea-01: Widen layer2 (8×8 stage) 256→384 toward airbench96 proportions

## Summary
Widen the 8×8 stage of `ResNet9` (train.py, lines 140-153) where cuDNN runs at full
speed, moving toward the airbench96 net's wider middle (128→384→512). Exact, complete
set of line changes (all in `ResNet9.__init__`):

- `self.layer2 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2), GatedResidual(256))`
  → `nn.Sequential(conv_bn(128, 384), nn.MaxPool2d(2), GatedResidual(384))`.
  Both the `conv_bn` out-channels AND the `GatedResidual` channel arg must change to 384
  so layer2's internal shapes stay consistent (GatedResidual is `c→c`).
- `self.layer3 = nn.Sequential(conv_bn(256, 512), nn.MaxPool2d(2), Residual(512))`
  → `conv_bn(384, 512)` — the layer3 stem's INPUT must match layer2's new 384-channel
  output. `Residual(512)` is unchanged; layer3 OUTPUT stays 512.
- `self.pool` (MaxPool2d(4)) and `self.fc = nn.Linear(512, num_classes, bias=False)`
  need NO change: layer3 still emits 512 channels, so the fc input dim stays 512.

No other line touches these shapes (`whiten`, `prep`, `layer1` all upstream of layer2).
This is the only configuration I propose as primary: **layer2 256→384, everything else
fixed** — a clean single-variable capacity test that mirrors EXP-004's logic at the same
proven 8×8 stage. Leave `PEAK_LR=0.4` UNCHANGED: the new GatedResidual(384) is still
identity-init (α=0, line 134) so the residual branch is inert at init and needs no LR
retune (EXP-004 confirmed ReZero needs none). The added kaiming-init conv_bn channels in
`layer2[0]` (128→384) and `layer3[0]` (384→512) are on the MAIN (non-residual) path, so
init signal scale shifts slightly there; kaiming is width-invariant for variance, so this
is benign and not a reason to retune LR.

## What it targets
The capacity limiter near the ~96.0% ceiling. EXP-004 (96.00%) showed capacity is still
binding at this scale: adding ONE block at layer2 lifted +0.13pp and its capacity lead
emerged by ep25 (92.63 vs 88.84) despite 142 vs 174 epochs. We fit ~142-150 epochs in
300s vs airbench96's 37 in ~35s — a generous epoch budget that can pay for wider layers,
and VRAM is a non-constraint (1.6 of 98 GB, EXP-001). airbench96 (96.03%) uses exactly
this wider middle (128→**384**→512), so the target width has external precedent.

## Reasoning
The 8×8 stage is the right place to spend capacity for two reasons. (1) Throughput:
layer2 convs run on 8×8 maps where cuDNN selects fast kernels — EXP-005 proved the
coarse 4×4 layer3 is BOTH capacity-saturated AND ~10% slower per FLOP (cuDNN
small-spatial penalty), so widening layer3 is explicitly excluded. (2) Precedent +
target: EXP-004's capacity add at exactly this stage outran its epoch cost, and airbench96
documents 384 as the working middle width. Widening (more channels, same depth) adds
representational width where EXP-004 already showed capacity pays, without adding the
extra sequential conv-latency that deepening incurs.

## Estimated effort
Low. ~3 line edits in one method; no schedule, optimizer, or data-pipeline changes.

## Risk assessment
- **Throughput → epoch loss.** layer2 conv FLOPs scale ∝ width² on the residual/main
  convs. Going 256→384 is 1.5×, so layer2's three 3×3 convs cost ~2.25× their FLOPs,
  and layer3's stem input grows 256→384 (1.5×). layer2/layer3 are a large share of total
  FLOPs, so expect a meaningful img/s drop and an epoch drop from ~142-150 toward perhaps
  ~115-130. EXP-004 tolerated 142 vs 174 and still won, so this is plausibly survivable,
  but it is the central capacity-vs-epochs tension and the main failure mode: if the wider
  net under-anneals (most gain is in the low-LR tail, EXP-001), the tail could fall below
  96.00%.
- **Noise floor.** The +0.1pp bar sits AT the ~0.1pp run-to-run noise floor (epoch-count
  jitter from the time-budgeted loop, seed fixed). The gain must clearly exceed 0.1pp to
  register; a sub-0.1pp single-run result is unproven. EXP-004's +0.13pp barely cleared
  this, so a 1.5× widen is a reasonable bet for a clearly-larger margin, but it is not
  guaranteed.
- **Clean test.** Single variable: change ONLY the four width args above (layer2 conv_bn
  out, GatedResidual, layer3 conv_bn in), hold PEAK_LR=0.4 and all else byte-identical to
  EXP-004. Compare best_test_acc vs 96.00, and log num_epochs/img_per_sec to attribute any
  shortfall to under-annealing vs capacity. If borderline, a smaller step (256→320) is the
  natural fallback to recover epochs.
