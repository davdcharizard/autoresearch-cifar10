# Brainstorm EXP-045
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **PyTorch `torch.compile` mode docs** (pytorch.org/docs — `torch.compile`, mode argument): `"max-autotune"` triggers an exhaustive kernel autotuning + epilogue-fusion search (incl. Triton matmul/conv templates and CUDA-graph capture) — slower to compile, potentially faster steady-state than `"reduce-overhead"`. Compilation is LAZY (on first forward call); a deliberate warmup call compiles the graph then, so the one-time compile cost can be moved OFF a per-step-timed budget by triggering it before the timed loop.
- **"Train longer, generalize better" / Ghost BatchNorm** (Hoffer et al., NeurIPS 2017): computing BN statistics over small "ghost" sub-batches (e.g. 32 of a 128 batch) injects stochastic-gradient-like noise into normalization → an implicit regularizer that improves generalization, most pronounced at large batch but documented at moderate batch too.
- **"Precise BN" / EvalNorm** (Singh & Shrivastava 2019; used in FAIR pycls): the running BN stats accumulated as an EMA over the trajectory lag the final weights; recomputing BN running mean/var with the FINAL weights over a few hundred forward-only batches better matches the eval distribution. Benefit is largest when end-of-training LR is high (stats lag); shrinks toward zero for cosine-to-0 schedules (final weights ≈ EMA-tracked weights).
- **project-insights.md** (local): EXP-040 corollary — "`torch.compile(reduce-overhead)` already reaches the CONV DT FLOOR; cheap throughput flags (cudnn.benchmark/TF32) can't buy epochs. Only max-autotune or moving work off the timed GPU path could cut dt. NOTE: no experiment has yet ADDED real epochs, so convergence-bound-vs-epoch-saturated at the ~91-ep/300s operating point stays OPEN."

## Experimental History Review

- **Current best / baseline**: 96.22% (EXP-012, commit 6c417a4), k=4 ResNet-20, ~91 epochs @ 8ms/step. **34 consecutive no-improvements** since EXP-012.
- **ALL accuracy axes closed**: capacity (now ALL THREE directions — width EXP-004/009, FLOP-neutral realloc EXP-038, depth EXP-044), augmentation (EXP-011/013/014/018/021/033-035/037), LR schedule (EXP-016/017/019/020/029), regularizer-adds (EXP-005/022), classifier head (EXP-032/039), intermediate-feature-routing (EXP-032/042), activations (EXP-010/028), weight-averaging (EXP-006/019/020), optimizer family + grad/objective mods (EXP-030/031/036/041/043), bag-of-tricks (EXP-026), large-batch (EXP-025), cheap-throughput flags (EXP-040).
- **The unifying constraint**: every accuracy lever is gated by the **epoch wall** (project-insights High) — adding compute/layers/penalties costs epochs → underfit. The plateau is generalization-bound at fixed capacity, and capacity is unmovable at 300s.
- **The ONE genuinely-open question** (EXP-040 corollary, repeated in EXP-044 Next Steps): is the *TrivialAugment* recipe convergence-bound or epoch-saturated at ~91 epochs? EXP-007's saturation finding (~77 ep) PREDATES TrivialAugment (EXP-012). Strong augmentation slows convergence and rewards longer schedules, so the post-TA net may still be improving at 91 epochs. **No experiment has ever genuinely ADDED net epochs to test this** — every throughput probe (EXP-007 compile, EXP-040 cudnn.benchmark) either spent its gain on a confound or hit the dt floor.
- **Untried**: (a) moving the one-time torch.compile cost OFF the per-step-timed budget via a warmup pass; (b) max-autotune kernel re-tuning; (c) the normalization axis entirely (Ghost BN); (d) end-of-training precise-BN recalibration.

## Candidate Ideas

### 1. Buy net-new epochs: compile-warmup off the timed budget + max-autotune
**Summary**: Two compounding changes whose SOLE purpose is to raise the realized epoch count at the byte-identical k=4 recipe, to test whether the TA-augmented net keeps improving past ~91 epochs. (a) **Compile-warmup**: before the timed training loop, run one forward+backward on a dummy `torch.zeros` batch through `compiled_model` (then `zero_grad`, do NOT step the optimizer, and reset BN running buffers) — this triggers `torch.compile` compilation during startup (charged to wall-clock, not to `total_training_time`), removing the ~14s one-time compile cost that is currently billed to the 300s budget on step 1 (EXP-007). Guaranteed ≈ +4-5 epochs. (b) **max-autotune**: switch `torch.compile(mode="reduce-overhead")` → `mode="max-autotune"` for a more aggressive kernel/epilogue search that MAY lower steady-state dt below the current 8ms floor (EXP-040 flagged this as the one untried dt reducer). The recipe (optimizer/schedule/aug/seed) is unchanged — the only thing that moves is epoch count, so any accuracy delta is cleanly attributable to "more epochs."

**Reasoning**: Directly resolves the single explicitly-open strategic question (EXP-040 corollary). The compile-warmup is a guaranteed-correct budget reclaim (the timer at train.py L218 starts after the dataloader yields and the compile fires on the first timed forward, so moving it earlier is pure budget savings with zero accuracy side-effect). If the TA net is convergence-bound, +epochs lifts top-1; if saturated, it definitively closes the "more epochs" question (currently OPEN). Either outcome is high-information.

**Sources**: PyTorch `torch.compile` mode docs; project-insights EXP-040 corollary + epoch-wall (High); knowledge/papers (EXP-007 compile entry); goal-learnings "More epochs / throughput scaling (epoch-saturated past ~77)" (EXP-007 — but pre-TA).

**Estimated Effort**: medium — warmup block + BN-buffer reset + mode string; must verify seed-safety (zeros input consumes no RNG; no optimizer step; BN buffers reset) and wall-clock budget.

**Risk Assessment**: (1) max-autotune compile can be slow (1-3 min) → risk of breaching the 10-min total wall limit → killed/crash. Mitigation: warmup moves compile to startup (wall, not budget), monitor wall, and fall back to reduce-overhead-mode warmup (still gains the ~14s) if max-autotune compile is excessive. (2) EXP-040 suggests the conv dt floor is already reached, so max-autotune may NOT cut steady dt — then the gain is only the ~+5 warmup epochs, possibly within the ±0.2pp epoch-jitter noise (goal-learnings Protocol) → inconclusive. (3) BN-buffer reset / warmup side-effects must not perturb the comparable run (seed hacking risk) — handled by zeros input + buffer reset + no step. Worst case: clean no-improvement that closes the saturation question.

### 2. Ghost BatchNorm — implicit regularization via small-sub-batch BN statistics
**Summary**: Replace `nn.BatchNorm2d` with a Ghost-BN that computes batch statistics over sub-groups of the 128 batch (e.g. ghost size 32 → 4 groups), sharing affine params and accumulating running stats normally (eval unchanged — uses population running stats). The noisier per-group statistics act as an implicit regularizer that can improve generalization at iso-compute.

**Reasoning**: The normalization axis is completely untouched across 45 experiments. Ghost BN is a genuine generalization lever (not convergence-slowing like dropout, not a loss-shape polish), so it sidesteps the "adding regularizers underfits" pattern in spirit. Compute-light.

**Sources**: Hoffer et al. 2017 (Ghost BN); contrast with EXP-022 dropout (penalty-style regularizer, underfit).

**Estimated Effort**: medium — custom GhostBN module replacing 18 BN layers + reshape logic.

**Risk Assessment**: Ghost BN's documented benefit is strongest at LARGE batch (>256); at batch 128 the extra stat-noise from ghost-32 is modest and may not help (or may add noise that hurts the already-well-conditioned net). HIGH torch.compile risk: per-block reshapes can break the reduce-overhead CUDA graph → dt doubles (the EXP-042 gotcha) → epoch wall → confounded. Uncertain benefit + real dt risk.

### 3. Precise-BN recalibration at end of training
**Summary**: After the timed loop, with final weights frozen, recompute every BN layer's running mean/var over a few hundred forward-only training batches (cheap, no backward), then evaluate. Better-matches eval-time BN stats to the final weights.

**Reasoning**: Cheap, untried, and targets the train/eval BN-stat mismatch directly.

**Sources**: Singh & Shrivastava 2019 (precise BN / EvalNorm); FAIR pycls.

**Estimated Effort**: low — a post-training BN-recalibration loop.

**Risk Assessment**: Benefit shrinks to ~zero for cosine-to-0 schedules — the final near-zero LR means the BN EMA already tracks the final weights closely (the SAME reason EMA was a no-op on cosine-to-0, EXP-006). Almost certainly a near-null here. Lowest expected impact; included for completeness.

## Idea Evaluation

The strategic context dominates: 34 no-improvements, every accuracy axis closed, all gated by the epoch wall. The highest-value move is not another accuracy lever (they're exhausted) but resolving the one open *structural* question — is the TA recipe convergence-bound at 91 epochs? Only #1 attacks that.

- **Evidence strength**: #1 rests on a concrete, mechanism-level project finding (EXP-040: compile fires inside the timed loop; conv floor reached but max-autotune untried; epochs never genuinely added) plus a guaranteed-correct budget reclaim (the warmup). #3 has clear literature but a project-specific reason to expect null (cosine-to-0, cf. EXP-006). #2 has literature but for a different regime (large batch) and a serious dt-confound risk on this compile-sensitive net.
- **Mechanism clarity**: #1 is the clearest — warmup moves a known ~14s cost off the budget (pure epochs); max-autotune may cut dt (more epochs). The accuracy comparison is single-variable (epoch count) because the recipe is untouched. #2/#3 have murkier paths to top-1.
- **Expected impact / risk**: #1 fails gracefully (worst case: a clean no-improvement that CLOSES the open saturation question — itself valuable) and has a guaranteed-correct component; its main risk (max-autotune wall-clock) is mitigated by a reduce-overhead fallback. #2 risks a dt-confounded regression (CUDA-graph break). #3 is likely null.
- **Feasibility**: #3 lowest-effort but lowest-impact; #1 medium-effort, highest strategic value; #2 medium-effort, uncertain.

#1 wins decisively: it is the only candidate that addresses the actual binding constraint (the epoch wall / saturation question) rather than poking another closed accuracy axis, and it is the move the prior analysis (EXP-040, EXP-044 Next Steps) explicitly identified as the remaining lever.

## Chosen Idea
**Selected**: Buy net-new epochs — compile-warmup off the timed budget + max-autotune

**Why this idea**:
After 34 consecutive no-improvements with every accuracy axis closed, the binding constraint is unambiguously the epoch wall, and the only open question is whether the TrivialAugment recipe (post-EXP-012, never tested for saturation — EXP-007's finding predates TA) is still convergence-bound at ~91 epochs. This is the one experiment that genuinely ADDS net epochs (no prior experiment has): the compile-warmup is a guaranteed-correct reclaim of the ~14s one-time compile cost currently billed to the 300s budget, and max-autotune is the single untried dt reducer (EXP-040). The recipe is byte-identical, so any accuracy delta is cleanly attributable to epoch count. It fails gracefully — a null definitively closes the saturation question, which is itself high-value after a long plateau.

**Hypothesis**:
Moving the torch.compile cost off the per-step-timed budget (warmup) plus max-autotune will raise realized epochs from ~91 to ≥100 (warmup alone ≈ +5; max-autotune, if it cuts dt below the 8ms floor, more). IF the TrivialAugment recipe is convergence-bound at 91 epochs, the extra epochs lift best_test_acc by ≥0.1pp over 96.22 (≥96.32). The null/regression case — epochs rise but best_test_acc stays within ±0.2pp of 96.22 — would be decisive evidence that the net is epoch-SATURATED even under strong augmentation, finally closing the last open question and confirming 96.22 as the fully-mapped k=4/300s ceiling. (Abort/fallback: if max-autotune compile threatens the 10-min wall, fall back to reduce-overhead-mode warmup, which still adds the ~5 epochs.)
