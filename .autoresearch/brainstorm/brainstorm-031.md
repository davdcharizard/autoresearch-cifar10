# Brainstorm EXP-031
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **Gradient Centralization (Yong et al., ECCV 2020)** (knowledge/papers/gradient-centralization.md)
  GC centralizes each weight gradient to zero-mean over fan-in per output unit; reported ~+0.2–0.6% on CIFAR/ImageNet ResNets via a 3-line drop-in between backward and step. Mechanism: constrain weight space + standardize gradient → BOTH accelerate convergence AND regularize. The math is FLOP-trivial; the only cost is the per-parameter reductions.
- **PyTorch performance — eager per-op overhead & `torch.compile`/Inductor fusion** (official PyTorch docs; pytorch.org/tutorials torch.compile, torch._foreach multi-tensor ops)
  In eager mode each tensor op carries Python + dispatcher + launch overhead (~10–20µs). A Python loop issuing ~46 tiny ops/step (23 mean + 23 sub) costs ~0.9ms of HOST time even though the GPU work is trivial — exactly the EXP-030 dt 8→9ms regression. Two standard remedies: (a) `torch.compile` the operation so the whole loop becomes one compiled artifact (fuses mean+subtract per tensor into a single triton kernel, issued from compiled code with ~no per-op Python overhead); (b) `torch._foreach_*` multi-tensor ops batch many same-shape ops into one launch — but they do NOT broadcast, so they can't directly do `grad - mean(grad)` where the mean is a reduced-shape tensor. Compile is the cleaner fit here.

## Experimental History Review

Current best **96.22%** (EXP-012, commit 6c417a4); bar = 96.32 (+0.1pp). 31 experiments; ~22 axes closed (scalar knobs bracketed, aug family closed, regularizer-adding fails, compute-adding hits the epoch wall, weight-averaging/polish moves loss-not-top1, downsampling closed both sides, activation closed both recipes, LR-schedule fully closed).

**THE LEAD — EXP-030 Gradient Centralization, the project's strongest near-miss** (TSV row 030, reports/exp-report-030.md, goal-learnings Patterns High):
- GC → **96.21** (−0.01pp = statistical TIE) with **final_test_loss 0.1934 < baseline 0.195 (BETTER)**.
- CRITICAL confound: the naive per-param Python for-loop (23 grad tensors × mean+sub ≈ 46 tiny eager kernel launches) cost **~1ms/step → dt 8→9ms → epochs 91→88 (−3)**. GC TIED baseline AND beat it on loss **DESPITE the 3-epoch handicap**. A throughput-confounded run normally REGRESSES (cf. EXP-024 BlurPool −0.56pp at 77 ep, EXP-028 SiLU −0.24pp at 88 ep); GC instead HELD → its per-epoch effect is mildly POSITIVE, masked by the loop's host overhead.
- **The optimizer/gradient-dynamics axis is OPEN** (previously only WD swept, EXP-005) — GC is the first lever to NOT lose to baseline.

**The actionable gap**: remove the throughput confound. EXP-030 lost 3 epochs purely to eager per-op Python/launch overhead in the GC loop — NOT to GC's math (which is trivial). Restoring GC to ~91 epochs / dt~8ms gives the FAIR test EXP-030 couldn't, and is the single most promising remaining shot at the 96.32 bar.

**Relevant project-insights**: (Medium, EXP-030) a per-PARAM Python loop running each step costs measurable dt via host overhead and can MASK the effect → vectorize/compile it; a throughput-confounded near-miss ⇒ re-test the SAME idea throughput-neutral, do NOT close the axis. (High, EXP-004/009/015/024) only compute-NEUTRAL changes are fairly testable at 300s — so the GC re-test MUST be verified throughput-neutral (epoch count) to be meaningful.

## Candidate Ideas

### 1. Throughput-optimized GC via a `torch.compile`d centralization over a pre-hoisted weight-grad list
**Summary**: Re-run EXP-030's Gradient Centralization but eliminate the per-step host overhead. Two changes, both in `train.py`: (a) **hoist** the weight-parameter list out of the training loop — compute `gc_params = [p for p in model.parameters() if p.ndim > 1]` ONCE before the `while` loop (the 23 conv/fc weights; skips iterating the ~40 BN γ/β + bias params and the `ndim` check + `tuple(range(...))` alloc every step); (b) **compile** the centralization: define a module-level `_gc(grads)` that loops `g.sub_(g.mean(dim=tuple(range(1,g.ndim)), keepdim=True))` and wrap it `_gc_c = torch.compile(_gc)`. Each step, between `loss.backward()` (L237) and `optimizer.step()` (L238), call `_gc_c([p.grad for p in gc_params])`. Inductor fuses each tensor's mean+subtract into a single triton kernel and issues all of them from compiled code with ~no per-op Python/dispatch overhead → the ~1ms/step eager cost collapses, restoring dt~8ms / ~91 epochs. Identical GC math to EXP-030 — only the execution is optimized. No config/param change.

**Reasoning**: EXP-030 proved GC's *math* is at worst neutral and probably mildly positive (tied baseline + better loss at a 3-epoch DISADVANTAGE). The ONLY thing standing between EXP-030 and a potential pass is the 3 lost epochs, and those came entirely from eager per-op host overhead in the GC loop — the textbook use case for `torch.compile` (this very net already gets +30% from `torch.compile(reduce-overhead)`, EXP-007). Hoisting removes per-step Python work; compiling removes per-op dispatch. If this restores ~91 epochs, GC gets its fair test: given it TIED at 88 epochs, at 91 it could edge past baseline and possibly approach/clear the 96.32 bar. Targets the real bottleneck (host launch overhead), not GC itself.

**Sources**: reports/exp-report-030.md (the near-miss + 3-epoch confound); knowledge/papers/gradient-centralization.md; project-insights EXP-030 (vectorize per-param loops) + EXP-007 (torch.compile is a proven ~30% lever on this launch-bound net); train.py L184 (param iteration), L213 (loop start = hoist point), L237-238 (GC insertion point), L240 (sync — dt captures GC).

**Estimated Effort**: low (hoist one list + a compiled 3-line function + one call site; one 300s run).

**Risk Assessment**: (a) **torch.compile + in-place grad mutation**: Dynamo supports input mutation in default mode, but if it balks/recompiles or silently clones, the win shrinks or it errors. MITIGATION: smoke-test the compiled GC on real grads first (correctness + runs clean); use DEFAULT compile mode (not reduce-overhead) to avoid CUDA-graph static-address pitfalls with external grad buffers. (b) **First-step compile cost**: the compiled GC traces on step 1, inside the timed region (~1–3s, charged to budget once) — negligible (<<1 epoch), note it. (c) **Residual overhead**: if compile only halves the launches (default mode issues fused kernels but not CUDA-graphed), dt may land ~8.4ms / ~89–90 ep — still a much fairer test than 88, and GC's positive lean may still show. (d) **Outcome floor**: even fully neutral, GC's gain may sit within the ±0.2pp noise floor on this shallow net → no-improvement, but it cleanly STAMPS the GC result as a fair tie/near-miss and informs whether to pursue other optimizer-class members. Compute-neutral target → no epoch-wall/integrity risk.

### 2. Hoist-only eager GC (precompute weight-param list + reduction dims; no compile)
**Summary**: The conservative subset of Idea 1: precompute `gc_specs = [(p, tuple(range(1, p.ndim))) for p in model.parameters() if p.ndim > 1]` once before the loop, then each step run the eager loop `for p, d in gc_specs: p.grad.sub_(p.grad.mean(dim=d, keepdim=True))`. No `torch.compile`. Removes the per-step Python overhead (iterating ~40 BN/bias params, the ndim check, the tuple allocation) but keeps the ~46 eager kernel launches.

**Reasoning**: Zero compile risk — guaranteed to run. Recovers the Python-side fraction of the EXP-030 overhead (the part from iterating all params + per-iter allocations), likely worth a fraction of an epoch. A safe fallback if compiling grad mutation proves fragile.

**Sources**: reports/exp-report-030.md; train.py L213/L237-238.

**Estimated Effort**: low (precompute one list; one run).

**Risk Assessment**: The dominant cost in EXP-030 was the ~46 per-op dispatches/launches, which this does NOT cut — so it may only recover ~0.5–1 epoch (dt ~8.7ms, ~89 ep), possibly still short of a clean fair test. Lower expected throughput recovery than Idea 1; kept as the low-risk fallback.

### 3. Adaptive Gradient Clipping (AGC, Brock et al. 2021) — a different optimizer-class member
**Summary**: Instead of centralizing, clip each weight's gradient when the per-unit grad-to-weight norm ratio exceeds a threshold λ (≈0.01–0.04): `g *= min(1, λ·‖W_unit‖/‖g_unit‖)`. Another compute-light, optimizer/gradient-class intervention (the now-open axis), targeting training stability/larger-LR tolerance.

**Reasoning**: Keeps breadth in the newly-opened optimizer/gradient-dynamics class in case GC's throughput can't be salvaged to a win. AGC enabled high-LR BN-free training in NFNets; on a BN net at a tuned LR its marginal benefit is uncertain.

**Sources**: Brock, De, Smith & Simonyan, "High-Performance Large-Scale Image Recognition Without Normalization" (ICLR 2021, arXiv:2102.06171); project-insights EXP-030 (optimizer axis open).

**Estimated Effort**: medium (per-unit norm ratios — same per-param-loop overhead problem as GC, so would also need hoisting/compile; an extra λ hyperparameter to guess).

**Risk Assessment**: Carries the SAME per-param-loop throughput cost as GC plus an untuned λ and a weaker mechanism fit (AGC's payoff is largest for BN-free / very-high-LR regimes, neither of which holds here). Strictly lower EV than finishing the GC story first. Deferred.

## Idea Evaluation

All three respect the hard constraints (train.py-only, no new deps, single GPU, ≤1 eval/epoch, no seed hacking) and stay in the now-open optimizer/gradient-dynamics class.

- **Evidence strength**: Idea 1 directly builds on the project's STRONGEST near-miss (EXP-030: GC tied baseline + improved loss at a 3-epoch handicap) and uses a proven lever on this exact net (`torch.compile` gave +30%, EXP-007) to remove the one identified confound. Strongest evidence by far. Idea 2 is the same idea with a weaker fix. Idea 3 is an unproven new member with an untuned knob.
- **Mechanism clarity**: Idea 1 is crisp — the regression was host launch overhead (~46 eager ops/step); compile collapses that to a fused compiled call → restores epochs → fair GC test. Idea 2 only removes the Python-iteration slice, leaving the launches. Idea 3's accuracy mechanism is murky on a tuned BN net.
- **Expected impact**: Idea 1 is the only candidate that could convert the proven near-miss into a pass — highest upside. Idea 2 may not fully restore throughput; Idea 3 is exploratory.
- **Risk profile**: Idea 2 is safest (no compile) but lowest upside; Idea 1's compile risk is mitigated by a smoke test + default mode and fails gracefully to no-improvement; Idea 3 risks a wasted loop on an untuned method.
- **Feasibility**: 1 and 2 are trivial; 3 is medium and would itself need the same throughput fix.

Idea 1 clearly leads: it attacks the single confound separating the project's best near-miss from a potential gain, using a lever already validated on this net, with a mitigated failure mode. Idea 2 is its built-in fallback (if compile proves fragile, drop to hoist-only). Idea 3 is deferred to a later loop if GC variants exhaust.

## Chosen Idea
**Selected**: Throughput-optimized GC via a `torch.compile`d centralization over a pre-hoisted weight-grad list

**Why this idea**:
EXP-030 is the project's strongest near-miss — Gradient Centralization TIED the 96.22 baseline AND improved test loss (0.1934<0.195) **despite a self-inflicted 3-epoch throughput handicap** caused entirely by the eager per-param GC loop's ~1ms/step host overhead. The GC math is FLOP-trivial; the lost epochs are a pure execution artifact. `torch.compile` (proven to give +30% on this launch-bound net, EXP-007) fuses the 23 per-tensor mean+subtract ops into one compiled call, collapsing the per-op Python/dispatch overhead and restoring dt~8ms / ~91 epochs. This delivers the fair, throughput-neutral test of GC that EXP-030 couldn't — and since GC already tied at a disadvantage, removing the disadvantage is the most promising remaining move toward the 96.32 bar.

**Hypothesis**:
Running GC through a pre-hoisted, `torch.compile`d centralization restores the epoch count to ~91 (dt~8ms, confirming the throughput fix) and lifts `best_test_acc` above EXP-030's 96.21 — toward or past the 96.32 bar — at unchanged 4,299,866 params / <600s. Falsifiable two ways: (1) if epochs stay ~88 (dt~9ms), compile failed to cut the overhead → the fix didn't take, re-approach; (2) if epochs recover to ~91 but accuracy lands within ±0.2pp of 96.22, GC's intrinsic per-epoch benefit is genuinely below the noise floor on this shallow well-tuned net → GC is a fair tie and the optimizer/gradient-dynamics axis needs a different member (AGC, grad-noise), not GC.
