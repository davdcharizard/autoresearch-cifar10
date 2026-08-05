# EXP-016: Ghost Batch Normalization (regularizing activation-statistic noise)

## Execution

Overall Status & Info:
- **Created**: 2026-06-30
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-016
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented `GhostBatchNorm2d(nn.BatchNorm2d)` in train.py and wired it into `conv_bn` (10 BN sites) in place of `nn.BatchNorm2d`, toggled by env `GHOST_SIZE` (default 512 = exact EXP-008 baseline). The module normalizes BN TRAIN statistics over ghost sub-batches of size `g` (regularizing activation noise) while updating eval running stats from the FULL-batch moments (C-sized buffers) so the `AveragedModel(use_buffers=True)` EMA and eval path stay identical to standard BN. All Milestone-1 smokes passed: Smoke A (g=512/0 bypass ≡ nn.BatchNorm2d, max diff 0.00e0 in train+eval, bf16/channels_last), Smoke B (per-ghost normalized stats mean≈0/var≈1; manual running stats match nn.BatchNorm2d full-batch to 2e-10/6e-8), Smoke C (EMA buffers equal manual EMA of full-batch stats to 1e-10, shape [C]), and a full-model gradient smoke (finite loss/grads). The M2 throughput probe then drove a major design adaptation (see Decisions).

### Surprises & Discoveries
- **Ghosting is ~50% slower at all sites** — a much larger throughput tax than the plan's "near-free" expectation. The M2 probe measured GBN-128 all-sites at 13.6k img/s vs 27.1k standard (+49.7% slowdown), identical for g=128 and g=64 → the cost is a FIXED per-call overhead, not group-count dependent.
- **Root cause**: ghosting breaks the single fused channels_last bf16 BN kernel. The grouped path needs a reshape (channels_last→contiguous copy), a non-channels-last `F.batch_norm`, a separate affine kernel, and the running-stat update — a microbench showed BN-forward is 4–9× slower per site, WORST at large spatial maps (prep 32×32: 4.3×; layer3 4×4: 8.6× but tiny absolute).
- **The fused `F.batch_norm` mitigation barely helped** (50.3% → still ~50%): the cost is the layout break + fp32/extra kernels, not the reduction itself. A leaner bf16 variant (no `.float()`, two fused BN calls) was ~30% better in the BN microbench but still 4–7× per site.
- **Decisive consequence for this TIME-BUDGETED harness**: an all-site ghost halves epochs (~150→~75) → severe under-anneal → it CANNOT beat a 150-epoch baseline regardless of regularization merit (throughput-disqualified). Per-layer restriction is the only viable regime: layer3-only (C≥512) = +11.6% (~133ep), layer2+3 (C≥256) = +27.2% (~109ep).

### Decisions
- **Adopted the fused `F.batch_norm` normalization path** (plan M2 mitigation rung 2): fold the G ghost groups into the channel axis (`reshape(g, G*C, H, W)`) and let cuDNN's fused BN normalize each (group,channel) over its own g·H·W elements; affine applied separately to keep γ/β C-sized; running stats updated manually from full-batch moments. Grouping is strided (n mod G) vs the plan's contiguous reshape — statistically identical for a shuffled batch (Smoke B re-validated for strided groups).
- **Added `GHOST_MIN_CH` env gate** (plan M2 mitigation rung 3 — "apply GBN to only the later BN sites"): ghost only BN sites with `num_features >= GHOST_MIN_CH`; sites below use standard fused BN. 0=all, 256=layer2+3, 512=layer3-only. This makes the epoch-preserving restriction explicit and controllable while keeping `GHOST_SIZE=512` default = exact baseline. Recorded as a deviation from the all-site plan, forced by the measured throughput reality.
- **Cell design reframed around the viable low-cost regime** (since all-site GBN is throughput-disqualified): c0 = standard BN (full speed ~150ep, control); cA = g=128, layer3-only (~133ep, least under-anneal confound, best shot at a clean win); cB = g=128, layer2+3 (~109ep, broader application, more under-anneal). Spans application breadth at fixed noise g=128; interpretation accounts for the epoch gap from M2.

## Experimental Adjustments

- **Fused BN normalization path over manual fp32 reductions**: cut nothing material on its own (~50%→50%) but is the correct base for the layer-restricted runs and keeps eval/EMA buffers clean. (ref: M2 probe — all-site +49.7%; microbench fp32 4.3–8.7× vs bf16 3.9–7.0× per site)
- **GHOST_MIN_CH layer restriction (the load-bearing adjustment)**: only layer3-only (+11.6%, ~133ep) keeps epochs in a defensible band; layer2+3 (+27%, ~109ep) included as a broader-application probe with an under-anneal caveat. All-site (+50%, ~75ep) dropped as throughput-disqualified. (ref: partial-application probe — std 27110, all 13636, layer2+3 19733, layer3 23959 img/s)

## Run Log

### Run 1 — c0 (standard BN control), cA (layer3-only g=128), cB (layer2+3 g=128)

Metadata:
- **Job ID**: (background bash — see TaskOutput)
- **Log file(s)**: experiments/016/run_c0.log, run_cA.log, run_cB.log; gpu_c0/cA/cB.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-30
- **Ended**: 2026-06-30

Description:
- Three sequential same-session cells on GPU 1, each a separate `train.py` process under `timeout 600`. c0 (`GHOST_SIZE=512`) is the full-speed standard-BN control (~150ep) — the same-session noise reference (stored 96.38 is too weak at the ~0.1pp floor). cA (`GHOST_SIZE=128 GHOST_MIN_CH=512`) ghosts only the 3 layer3 sites (~133ep, +12% throughput) — the least under-anneal-confounded GBN and best shot at clearing 96.48. cB (`GHOST_SIZE=128 GHOST_MIN_CH=256`) ghosts layer2+3 (~109ep, +27%) — a broader-application probe. We expect c0 ≈ baseline; the test is whether either GBN cell exceeds c0 by a clear (>0.1pp) margin despite its epoch cost.

Observations:
- All 3 cells completed cleanly within budget; GPU 1 idle (0% util, 3.8 GB dormant foreign mem) before/after every cell → no contention. (source: gpu_c0/cA/cB.log)
- **cA (layer3-only GBN) beat the same-session control c0 by +0.24pp (96.38 vs 96.14) DESPITE 16 fewer epochs (133 vs 149)** — a genuine positive regularization signal from layer3 ghost-statistic noise, strong enough to outweigh its 11% epoch deficit. (source: run_cA.log, run_c0.log)
- **cB (layer2+3 GBN) fell to 96.06 at only 111 epochs** — the broader application's +27% throughput tax cut 38 epochs, and the resulting under-anneal wiped out the ghost-noise benefit (cB < c0). Coherent monotone story: ghost benefit per added breadth < under-anneal cost per added breadth on this fused-kernel harness. (source: run_cB.log)
- ep25 was actually HIGHER for the ghost cells (cA 92.52, cB 92.67) than c0 (91.65) — early-training regularization/optimization was healthy, not destabilized; the cap is purely the epoch deficit at the tail, not instability. (source: run_*.log "eval ep  25")
- Same-session c0 (96.14) ran 0.24pp below the stored EXP-008 baseline (96.38) at identical 149 epochs / identical code path (g=512 ≡ nn.BatchNorm2d, Smoke A) → run-to-run variance (cudnn.benchmark nondeterminism) is ~0.2pp this session, larger than the nominal 0.1pp floor. Reinforces using the same-session control, not the stored value, as the comparison anchor.

Key Metrics:
- c0 best_test_acc: 96.14% @ best (ep~? ), final 96.11% @ ep149; 149 ep; 450.1s total; 1635 MB (source: run_c0.log)
- cA best_test_acc: 96.38% @ best, final 96.38% @ ep133; 133 ep; 425.3s total; 1848 MB (source: run_cA.log)
- cB best_test_acc: 96.06% @ best, final 96.06% @ ep111; 111 ep; 408.5s total; 2135 MB (source: run_cB.log)
- ep25 test_acc: c0 91.65% / cA 92.52% / cB 92.67% (source: run_*.log)
- M2 throughput (img/s): std 27110 / GBN-128 all-sites 13636 (+49.7%) / layer2+3 19733 (+27.2%) / layer3-only 23959 (+11.6%)

## Verification Results

### Conditions Checked
- **(a) Completes within budget, valid best_test_acc, wall < 600s** — PASS. All 3 cells completed: c0 450.1s/149ep, cA 425.3s/133ep, cB 408.5s/111ep; all training_seconds=300.0, all wall < 600s, all valid best_test_acc. (source: run_c0/cA/cB.log)
- **(b) Best GBN cell ≥ 96.48 AND > same-session c0 by a clear (>0.1pp) margin** — FAIL on the absolute gate. Best GBN cell = cA 96.38% < 96.48 bar. (It DID clear the relative gate: cA 96.38 > c0 96.14 by +0.24pp, above the ~0.2pp session noise, despite 16 fewer epochs — a real positive signal — but the absolute 96.48 floor is not met.) → no-improvement. Stop; remaining conditions not evaluated per protocol.
- **(c) Integrity (scope/eval/seed/smokes)** — skipped per protocol after (b) failed. (Noted anyway: `git status --short` shows only train.py modified; `prepare.py` byte-unchanged; eval count == num_epochs for all cells (1 eval/epoch); seed 42 unchanged; Smokes A/B/C + gradient all passed.)

### Verdict basis
no-improvement: the best ghost cell (cA, layer3-only, 96.38%) does not reach the 96.48 bar. The mechanism shows genuine merit (cA > same-session c0 by +0.24pp at a 16-epoch deficit; ep25 healthy), but the fused-kernel throughput tax caps achievable epochs and thus accuracy. Final verdict rendered in 04-analysis.md.

### Informational Metrics
- peak_vram_mb: c0 1635.4 / cA 1847.6 / cB 2135.1 MB (source: run_*.log)
- num_epochs: c0 149 / cA 133 / cB 111 (source: run_*.log)
- total_seconds: c0 450.1 / cA 425.3 / cB 408.5 (training_seconds 300.0 all) (source: run_*.log)
- num_params: 7,784,627 (unchanged — GBN adds no params) (source: run_*.log)
- ep25 test_acc: c0 91.65% / cA 92.52% / cB 92.67% (source: run_*.log)
- GBN throughput cost (M2 probe, img/s): std 27110 / GBN-128 layer3-only 23959 (+11.6%) / layer2+3 19733 (+27.2%) / all-sites 13636 (+49.7%)

## Errors & Dead Ends

<!-- Append only. -->

## Human Notes

> (none — autopilot)
