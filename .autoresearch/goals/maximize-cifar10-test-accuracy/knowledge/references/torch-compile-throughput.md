# torch.compile (TorchInductor) as a throughput lever under a fixed TIME budget

**Why it matters here**: At a fixed 300s TRAINING budget, throughput sets the epoch count, which gates
both anneal completion and realizable capacity. `torch.compile` is the rare lever that ADDS throughput
(more epochs) rather than spending it — the meta-lever behind the EXP-005/007/013 under-anneal wall.

**Mechanism**: Inductor captures the forward+backward graph and fuses elementwise/normalization ops into
Triton kernels. On a conv net the convs already dispatch to cuDNN (with `cudnn.benchmark=True`), so the
win is the GLUE: BatchNorm affine + ReLU (10 conv_bn blocks), residual adds (`Residual`, `GatedResidual`
ReZero scalar-mul + add), `flatten`/`*scale_out`, and the many bf16↔fp32 autocast casts. These are
bandwidth-bound pointwise/reduction kernels that fuse well, collapsing launch overhead + intermediate
round-trips. Static shape (batch 512, `drop_last=True`) → ONE compiled graph, no in-loop recompile.

**Grounded magnitude**: airbench "94% on CIFAR-10 in 3.29s" (arXiv:2404.00498) got **~14% throughput** from
torch.compile on a near-identical fast-CIFAR ResNet, "mathematically equivalent up to small float
differences," and notes the one-time multi-minute compile is only worth it amortized over many steps.
General small-CIFAR-convnet reports: 7–14% from compile alone (`mode="reduce-overhead"`/default for
training; `max-autotune` for max throughput at higher compile cost). On an already-fast small net expect
the LOW end (~7–15%); the elementwise share of wall time is modest.

**The off-budget compile-warmup pattern (critical for this goal)**: the budget accumulates per-step wall
time INSIDE the loop (`total_training_time += dt`), and anything before `t_start_training` is off-budget
(like the whitening eigendecomp). So pay the one-time compilation OFF-BUDGET: BEFORE `t_start_training`,
run a few warmup forward+backward passes at the EXACT static train shape (512,3,32,32) channels_last/bf16
in `model.train()` mode (compiles BOTH graphs). The timed loop then runs already-compiled at full speed.

**Correctness traps (hand-verified design)**:
- **BN running-stat pollution**: warmup forwards on dummy data mutate BN `running_mean/var`. SNAPSHOT all
  BN buffers before warmup, RESTORE after. Warmup does NO `optimizer.step()` (params/momentum stay clean);
  zero grads after.
- **Eval recompile vs the 10-min wall**: do NOT rebind `model`. Keep the ORIGINAL `model` uncompiled for
  all eval/EMA; use a separate `train_fwd = torch.compile(model)` ONLY for the training forward. The
  `AveragedModel(use_buffers=True)` EMA copy is already uncompiled → zero eval recompile. (Eval is
  off-training-budget but still under the 600s wall cap.)
- **Optimizer/EMA param aliasing**: `torch.compile` returns an `OptimizedModule` sharing the SAME tensors
  via `_orig_mod` (torch 2.9.1) — constructing optimizer/EMA over `model.parameters()` BEFORE compile is
  fine; `optimizer.step()` updates the tensors the compiled forward reads. Smoke-test this invariant.
- **Local dummy RNG**: generate warmup dummies with a LOCAL `torch.Generator` so global seed state (the
  fixed `manual_seed(42)`) is untouched — mirrors the whitening local-RNG discipline.

**Spend the bought throughput**: compile-only past ~150 epochs may be sub-noise near the ~96.4 ceiling;
the high-upside use is to OFFSET a mild capacity add (layer2 256→320, ~1.25× cost) so the added capacity
anneals — directly resolving the EXP-007 under-anneal failure. First tested EXP-014.

**Status (EXP-014, no-improvement but lever CONFIRMED)**: torch.compile (default mode, off-budget warmup)
delivered a clean, math-equivalent **+12% throughput** (154→173 epochs) — the off-budget-warmup + separate-
`train_fwd`-handle design worked exactly (warmup_seconds ~12s, no in-loop recompile, BN-restore/aliasing/
eval-boundary all verified). BUT it did NOT raise accuracy: +19 anneal epochs gave +0.03pp (anneal-saturated),
and compile-funded layer2-320 capacity annealed at a healthy 143 ep yet LOST −0.08pp (capacity-saturated).
Conclusion: this net is at a generalization ceiling, NOT epoch-bound — so compile is NOT a standalone accuracy
lever here. It IS worth BANKING as free +12% headroom to fund a DIFFERENT base architecture's per-step cost in
a future loop (the reusable recipe above is validated). See `experiments/014/04-analysis.md`.

Sources: arXiv:2404.00498 (airbench, §torch.compile); PyTorch torch.compile / TorchInductor docs; NVIDIA
Nsight + PyTorch 2.0 training-speedup blog.
