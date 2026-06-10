# Report EXP-030: Gradient Centralization (GC) on the SGD step

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-030.md
- **Plan**: plans/plan-030.md
- **Log**: logs/exp-log-030.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) by editing only `train.py` within the fixed 300s training-compute budget on a single H20. Baseline = **96.22** (EXP-012, commit 6c417a4); success bar = **96.32** (baseline + 0.1pp). This experiment tested whether adding Gradient Centralization to the SGD update lifts accuracy above the bar.

## Idea & Hypothesis
**Chosen idea**: Gradient Centralization (GC) — between `loss.backward()` and `optimizer.step()`, centralize every `ndim>1` weight gradient by subtracting its per-output-unit mean over the fan-in dims (conv: over (C_in,kH,kW) per C_out; linear: per row). 1-D params (BN γ/β, biases) untouched.

**Why selected**: After 30 experiments closing ~22 axes, the **optimizer/gradient-dynamics class** was the single genuinely-unexplored lever (only WD ever swept, EXP-005). GC (Yong et al., ECCV 2020, arXiv:2004.01461) is its best-documented, compute-neutral member — reported ~+0.2–0.6% on CIFAR/ImageNet ResNets via a 3-line drop-in. Its mechanism (constrain the weight space + standardize the gradient) targets BOTH binding constraints of this recipe at once — convergence-bound in the fixed budget AND the generalization gap at fixed capacity — without adding compute (no epoch wall) and without being a convergence-COSTING regularizer (the class that has repeatedly failed here).

**Hypothesis**: GC lifts `best_test_acc` above 96.32 at an unchanged ~91 epochs / 4,299,866 params / dt~8ms / <600s. Falsifiable: if epochs hold (~91) but accuracy lands within ±0.2pp of 96.22, GC's benefit is below the noise floor on this shallow well-tuned net.

## Approach
Single code change in `train.py`: a 4-line loop inserted between `loss.backward()` (L237) and `optimizer.step()` (L238):
```python
for p in model.parameters():
    g = p.grad
    if g is not None and g.ndim > 1:
        g.add_(-g.mean(dim=tuple(range(1, g.ndim)), keepdim=True))
```
No configuration change (PEAK_LR 0.2, WARMUP_FRAC 0.05, batch 128, WD 1e-4, LS 0.1, Cutout 16, TrivialAugment, Nesterov, momentum 0.9, cosine-to-0, seed 42 all unchanged). GC runs eager (outside the compiled forward region); the loop iterates the eager `model.parameters()` handle whose tensors share grads with the compiled model. In-place `add_` to avoid allocation. Smoke test confirmed 23 conv/linear weight tensors centralized (max |per-output-unit mean| after GC = 1.6e-8 ≈ 0), BN/1-D grads untouched, params 4,299,866 unchanged.

## Execution
One run, exit 0, clean compile, no NaN/Traceback. Loss descended normally. total_seconds 405.3 (<600). No retries, no adjustments. **Key execution surprise**: the un-fused Python GC loop (23 param tensors × mean+sub ≈ 46 tiny kernel launches per step) cost ~1ms/step → dt 8→9ms steady (678/679 sampled lines), dropping the epoch count from ~91 to **88** (−3). This was a launch-overhead throughput cost, not the expected compute-neutral behavior — the per-step reductions did not fuse away.

## Results
- **Primary metric**: best_test_acc **96.21%** (baseline 96.22, delta **−0.01pp**, −0.01%)
- **Observations**: final_test_loss **0.1934 — BETTER than baseline 0.195**, despite 3 fewer epochs. num_epochs 88, num_steps 33953, dt~9ms, peak_vram_mb 453.8. The −0.01pp top-1 is pure noise (statistical tie — the closest result of the session).
- **Analysis**: This is the **strongest near-miss of the project**. GC matched baseline top-1 AND beat it on loss **at a 3-epoch disadvantage** caused entirely by the GC loop's launch overhead (NOT by GC's gradient effect). A throughput-confounded experiment normally regresses (cf. EXP-024/028 epoch-wall losses); GC instead *held*. The natural reading: GC's intrinsic per-epoch effect is mildly POSITIVE, and that signal is being masked by the un-fused loop eating ~3 epochs of training. A throughput-NEUTRAL GC (vectorized loop restoring dt~8ms / 91 epochs) is the obvious follow-up and the best remaining lead toward the bar. The optimizer/gradient-dynamics axis is therefore OPENED, not closed.
- **Key Learning**: GC is the first lever to match baseline accuracy AND improve loss despite a self-inflicted epoch handicap — implying a real small positive effect masked by the un-fused loop's ~1ms/step overhead; vectorizing it (torch._foreach_) is the lead.

## Verification
- **Conditions**: Cond 1 (primary metric ≥ 96.32) **FAILED** (96.21 < 96.32). Cond 2 (clean completion <600s) and Cond 3 (no constraint violations: train.py-only, params unchanged, ≤1 eval/epoch, no new deps, seed unchanged) both PASS (informational).
- **Review Notes**: Results trustworthy. best_test_acc parsed from the summary block; git diff = train.py only; num_params 4,299,866 unchanged; pure torch ops (no new dep); no seed hacking. The throughput cost is honestly attributed (88 ep / 9ms, above the ~85-epoch fairness floor) so the accuracy delta is a *roughly* fair — and conservative — test of GC's merit.
- **Verdict**: **no-improvement**
- **Verdict Basis**: Cond 1 verification condition failed (96.21 < 96.32 bar); all hard constraints satisfied → no-improvement, not invalid.

## Unexplored Avenues
- **Throughput-neutral / fused GC (HIGH confidence lead)**: replace the per-param Python for-loop with a vectorized multi-tensor implementation — `torch._foreach_sub_` over the list of centralization offsets (each offset = `g.mean(dim=fan_in_dims, keepdim=True)` computed per-tensor, but the in-place subtract batched), or precompute the conv/linear param list once outside the loop to cut Python overhead. Goal: restore dt~8ms / 91 epochs so GC's intrinsic effect is tested without the 3-epoch handicap. Since GC tied baseline AND improved loss AT a disadvantage, removing the disadvantage is the single most promising untried move in the project.
- **GC restricted to conv layers only** (skip the final fc): smaller per-step op count → less overhead, and the paper notes conv-layer GC carries most of the benefit; a cheaper variant that may stay closer to throughput-neutral.
- **GC + slightly higher LR**: GC is reported to enlarge the stable LR range; a modest peak-LR bump (e.g. 0.2→0.24) co-tuned with GC could convert GC's landscape-smoothing into a convergence gain — but LR-peak is bracketed (EXP-016/017), so low marginal confidence.

## Next Steps
1. **EXP-031 — vectorized/fused Gradient Centralization (HIGH)**: re-run GC with the per-param Python loop replaced by a multi-tensor/`_foreach_` implementation (or a hoisted precomputed weight-param list) to eliminate the ~1ms/step overhead and restore 91 epochs / dt~8ms. This directly tests whether GC's masked positive signal clears or approaches the 96.32 bar. Strongest remaining lead.
2. **EXP-032 — conv-only GC (MEDIUM)**: if full-net fused GC still carries overhead, test conv-layer-only GC (fewer tensors, most of the documented benefit) as a cheaper variant.
3. Optimizer/gradient-dynamics class remains OPEN beyond GC: AGC (adaptive gradient clipping), gradient noise, or LARS-style layer-wise normalization are untried members for later loops if GC variants exhaust.

## Exit Action Results
<!-- No exit actions defined in the goal file. -->
- N/A — goal file defines no exit actions.
