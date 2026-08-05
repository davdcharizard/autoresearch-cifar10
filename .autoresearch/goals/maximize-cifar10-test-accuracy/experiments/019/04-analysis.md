# Report EXP-019: Squeeze-Excitation channel attention (layer2+layer3 residual branches)
- **Created**: 2026-06-30

## Goal
Maximize CIFAR-10 `best_test_acc` (%) within the fixed 300s training budget, editing only `train.py`. Direction: higher is better. Baseline: **96.38** (EXP-008, commit 07c3760). Improvement bar: best_test_acc ≥ **96.48** (baseline + 0.1pp) AND, given the ~0.1pp host-draw noise floor, clearly above a same-session control. This experiment tested whether adding Squeeze-Excitation (SE) channel attention — the first channel-attention lever on this goal — lifts the metric.

## Idea & Hypothesis
Chosen idea (Codex reviewer pick, 6.5/10, over schedule-shape and ConcatPool): **SE channel attention** in the residual branches. After 13 straight nulls and a robust generalization-ceiling diagnosis (EXP-014 disproved the epoch/throughput/capacity framing; EXP-016/017/018 closed BN-noise and the downsampling inductive bias), SE was the strongest remaining lever that adds a *genuinely new functional form* — content-adaptive per-channel recalibration conditioned on global image content — orthogonal to every saturated axis, at <1% params and near-zero compute (throughput-neutral, avoiding the under-anneal trap behind 5+ prior nulls). Hypothesis: SE (r=16) at the layer2/3 residual branches lifts best_test_acc ≥96.48 over the same-session control by a clear >0.1pp margin at near-full epochs, replicated on a confirmation re-run.

## Approach
All changes in `train.py` (sole editable file). Added a `class SE(nn.Module)`: GAP `x.mean((2,3),keepdim=True)` → 1×1 conv `c→max(8,c/r)` → ReLU → 1×1 conv `→c` → `2*sigmoid` gate → per-channel rescale. **Identity-init** the gate via zero-init `fc2` so the gate = `2·sigmoid(0)` = 1.0 at init (the residual branch is bit-unperturbed at init, critical for the un-ReZero'd `Residual(128)`/`Residual(512)` blocks; folded from the brainstorm review). SE threaded into `Residual`/`GatedResidual` after `c2` (inside the α-gate for the ReZero block). `se_layers`/`se_ratio` exposed as `ResNet9` constructor args defaulting to env globals `SE_LAYERS`/`SE_RATIO`, so in-process smokes build variants without env juggling. The load-bearing identity-init is a post-`self.apply()` loop re-zeroing every `SE.fc2` (kaiming from `apply` would otherwise clobber it).

Five plan-review concerns were folded before execution: cB demoted to diagnostic-only (verdict keyed on cA, no placement-search on the test metric); model-level identity smoke; `/tmp` helper scripts (train.py-only integrity preserved); constructor-arg config; background `nvidia-smi` contention sampling. Four smokes (model-level identity, baseline regression, param/structure, finite fwd/bwd) all passed; a throughput probe predicted cA 0.97×, cB 0.95× baseline (both clear of the num_epochs≥135 gate).

## Execution
Two same-session sessions on GPU 1, all clean (no foreign contention either session; GPU-1 mem all ours, GPU-0 util always 0). Session 1: c0/cA/cB. Session 2 (confirmation): fresh c0b/cAb. No retries, no errors. All 5 runs fully annealed at num_epochs ≥ 135, `best_test_acc == per-epoch max` (anti-gaming integrity), prepare.py byte-unchanged, only train.py modified.

The confirmation pair was an implementation-time decision: session 1's c0 drew low (96.11, ~0.27pp under the stored baseline), so cA's +0.28pp same-session lead, while above the noise floor, was potentially a low-c0-draw artifact (the symmetric case to the EXP-016/017 lesson). The pair tested replication and whether a normal host draw would clear 96.48.

## Results
- **Primary metric**: cA (SE layer2+3) **96.39%** (baseline 96.38, delta **+0.01pp**, +0.01%). Below the 96.48 bar.
- **Same-session pairs**:
  - Session 1: c0 96.11 / cA 96.39 (**+0.28pp**) / cB 96.21 (+0.10pp), epochs 149/144/139.
  - Session 2: c0b 96.29 / cAb 96.31 (**+0.02pp**), epochs 150/144.
- **Observations**: SE was throughput-neutral as designed (cA ~0.97×, 144 ep; the per-block GAP did NOT cause a CUDA sync stall — `x.mean` is an async reduction). Identity-init worked exactly (every SE block bit-identical to input at init; ep25 ran AT/ABOVE c0, no early-convergence depression). Diagnostic: cB (all-3 SE) 96.21 < cA (layer2+3) 96.39 — adding layer1 SE is net-negative (early 16×16/128-ch channel attention hurts, and cB also loses ~5 epochs), so layer2+3 is the better placement (vindicating the chosen hypothesis), but it still ties overall.
- **Analysis**: The hypothesis is **not supported**. SE achieved its intended local effect (a working, trainable, identity-initialized channel-attention mechanism at full epochs) but did not move the annealed optimum: the +0.28pp session-1 lead did NOT replicate (+0.02pp on confirmation), and cA's absolute (96.39, 96.31) never cleared 96.48. Averaged across two same-session pairs the SE effect is ≈+0.15pp but non-replicating and within the noise floor — i.e. no reliable signal. Session 1's c0 (96.11) was confirmed a low host draw (c0b drew 96.29), so the apparent win was largely a control artifact. Content-adaptive channel recalibration is **redundant** with the existing representation on this 7.8M-param heavily-augmented whitened ResNet-9 + EMA at 300s — consistent with the reviewer's honest modest-EV prior (ImageNet-scale SE gains do not transfer to this small, saturated CIFAR net).
- **Key Learning**: SE channel attention (the first channel-attention lever) ties the same-session control — a strong session-1 +0.28pp did not replicate (+0.02pp on confirmation) and never cleared 96.48 — so adding a new functional form (content-adaptive per-channel gating) is redundant on this saturated net; the generalization ceiling now resists architectural attention as well as capacity/optimizer/regularization/BN-noise/downsampling.

## Verification
- **Conditions**: condition 3 (primary metric — cA ≥ 96.48 AND cA−c0 > 0.1pp, replicated) **FAILED** (cA 96.39/96.31 < 96.48; +0.28pp → +0.02pp non-replicating). Conditions 1 (completion/budget), 2 (num_epochs≥135 + equal contention), 5 (ep25 sanity), 6 (integrity/anti-gaming) all PASSED.
- **Review Notes**: Results confirmed trustworthy — clean sessions, no contention, fully annealed, integrity intact, summary best == per-epoch max for all 5 runs. The two-pair design correctly exposed the session-1 lead as a low-c0-draw artifact (no false positive).
- **Verdict**: **no-improvement**.
- **Verdict Basis**: valid result, primary necessary condition failed (metric did not clear baseline+0.1pp and the same-session advantage did not replicate). No hard-constraint violation, no crash.

## Unexplored Avenues
- **SE at a different reduction ratio / placement variant** (r=8, or SE only at layer3): low EV — cB already showed adding breadth (layer1) hurts and SE params are negligible, so ratio is unlikely to convert a tie into >0.1pp; the layer2+3 r=16 operating point tested is already the best of the swept placements.
- **SE composed with a future architectural winner** (e.g. as a free rider on a different backbone): the mechanism is throughput-neutral and identity-initialized, so it composes cleanly — but on THIS backbone it is redundant. Worth re-testing only on a materially different feature extractor where the channel statistics are less saturated.
- **Other channel-recalibration forms** (CBAM spatial+channel, ECA parameter-free): same functional family; given SE ties here, these are very likely also sub-noise on this net — deprioritize.

## Next Steps
1. **Schedule-shape (cosine anneal, idea-02 with the `tail`→0 fix)** — throughput-free, EXP-012-flagged as an untried lever with ceiling above noise; the one cheap genuinely-different-axis probe left within this backbone. Confidence: low-medium (most-likely-to-tie, but zero-cost and never pulled).
2. **A wholesale different backbone** (the standing high-EV move after 14 straight within-DavidNet nulls) — e.g. a pre-activation/wider-stage ResNet or a small ViT/ConvMixer, funded by the banked torch.compile +12% throughput headroom to keep epochs in band. Confidence: medium (the ceiling is now robust to every within-architecture lever incl. attention; the diagnosis points off-architecture). This is the strongest remaining direction.
3. **AdaptiveConcatPool readout head (idea-03)** — cheapest remaining throughput-free probe; low confidence (reviewer 4.5/10, readout over a saturated representation), best folded as a rider on (2).
