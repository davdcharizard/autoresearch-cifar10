# Proposal 014-01: torch.compile (TorchInductor) for throughput → more in-budget anneal epochs (off-budget compile warmup)

## Summary
Wrap the training model in `torch.compile` and pay the one-time compilation cost OFF-BUDGET via a warmup forward+backward at the exact static train shape `(512,3,32,32)` channels_last/bf16, placed BEFORE `t_start_training`. The timed loop then runs an already-compiled model, raising img/s. Spend the bought throughput either on more anneal epochs (cell-A) or — the higher-upside framing — to offset the epoch cost of a mild capacity step (layer2 256→320, cell-B) so the added capacity actually anneals, directly attacking the EXP-007 under-anneal root cause. Same-session no-compile baseline (cell-0) is mandatory.

## Mechanism / Reasoning
The diagnosis (03-experiment-learnings) is unambiguous: at a fixed 300s TIME budget the net is throughput-bound and regularization-bound; every epoch-costing change under-anneals (EXP-005/007/013). torch.compile is the rare lever that *adds* throughput rather than spending it. The convs already dispatch to cuDNN (with `cudnn.benchmark=True`), so Inductor's win here is NOT the convs — it is fusing the elementwise/normalization glue into Triton kernels: BatchNorm2d affine + ReLU (10 `conv_bn` blocks), the residual adds in `Residual.forward` (`x + ...`) and `GatedResidual.forward` (`x + self.alpha * ...`, a scalar-broadcast multiply + add), the `flatten`/`* scale_out`, plus the many autocast bf16↔fp32 cast ops inserted around each op. These are memory-bandwidth-bound pointwise/reduction kernels that fuse well, collapsing kernel-launch overhead and intermediate round-trips. The static shape (batch 512, `drop_last=True`) means Inductor compiles ONE graph and never recompiles in the loop.

Honest magnitude: airbench (arXiv:2404.00498) reports torch.compile gave a ~14% throughput reduction on a near-identical fast-CIFAR ResNet, "mathematically equivalent up to small float differences." On an already-fast small net the elementwise share of wall time is modest, so I expect the LOW end: ~7–15% img/s. At baseline ~26k img/s / 150 epochs, +10% → ~165 epochs (cell-A), or it offsets a ~1.25× capacity step to hold ~150 epochs WITH more capacity (cell-B).

## Concrete implementation sketch (train.py-specific)
All edits in `train.py::main`, gated by env vars (EXP-012/013 multi-cell env-toggle pattern) so one process runs cell-0/A/B back-to-back with clean attribution.

1. `import os` at top. Read `USE_COMPILE = os.environ.get("USE_COMPILE","0")=="1"`, `COMPILE_MODE = os.environ.get("COMPILE_MODE","default")`, `LAYER2_WIDTH = int(os.environ.get("LAYER2_WIDTH","256"))` (cell-B only).
2. Compile the TRAIN model only, after model + whitening load + optimizer + EMA construction, but BEFORE `t_start_training` (line 268). Compile `model` in place:
   ```python
   if USE_COMPILE:
       model = torch.compile(model, mode=COMPILE_MODE)
   ```
   Critically, construct `optimizer` and `ema_model` over the ORIGINAL `model.parameters()` first (lines 244–257 already do), THEN reassign `model = torch.compile(model)`. `torch.compile` returns an `OptimizedModule` that shares the SAME underlying `nn.Module` and parameter tensors — so the optimizer still steps the exact params the compiled forward uses, and `ema_model.update_parameters(model)` reads `model.parameters()` which (via the wrapper's `_orig_mod` delegation) are the same tensors. No parameter aliasing break.
3. Off-budget warmup loop (also before `t_start_training`):
   ```python
   if USE_COMPILE:
       model.train()
       dummy = torch.randn(BATCH_SIZE,3,32,32, device=device).to(memory_format=torch.channels_last)
       dtgt = torch.randint(0,NUM_CLASSES,(BATCH_SIZE,),device=device)
       for _ in range(3):
           optimizer.zero_grad(set_to_none=True)
           with torch.autocast("cuda", dtype=torch.bfloat16):
               loss = criterion(model(dummy), dtgt)
           loss.backward()
       optimizer.zero_grad(set_to_none=True)  # discard warmup grads
       torch.cuda.synchronize()
   ```
   This compiles BOTH forward and backward graphs at the exact loop shape. We do NOT call `optimizer.step()` (keeps momentum buffers pristine) and zero grads after. `manual_seed(42)` is set at line 199 before this — re-seed is unnecessary since warmup only triggers compilation; but to be safe against any RNG consumption (dropout-free here, so none), the loop's data RNG is from the DataLoader, untouched by `torch.randn`/`torch.randint` on the default generator. NOTE: `torch.randn` advances global RNG — to preserve bit-identical seed behavior vs cell-0, generate dummies with a LOCAL generator (`torch.Generator(device).manual_seed(0)`), mirroring the whitening code's local-RNG discipline (line 85).
4. EVAL path stays uncompiled and correct for free: `ema_model` is the `AveragedModel` wrapper (line 255), NOT the compiled `model`. Once EMA starts (the normal case for best_acc), `eval_target = ema_model` (line 345) → eval runs the uncompiled EMA copy → ZERO eval recompilation, regardless of eval's dynamic last-batch shape (10000/256 → last batch 16). The compiled `model` is only eval'd in the pre-warmup epochs (`eval_target = model`, line 348); to avoid an off-budget eval recompile at dynamic shapes there, those early epochs are few and off-budget (within the 10-min wall) — acceptable.
5. cell-B only: parametrize layer2 width. Change line 150 to use `LAYER2_WIDTH` for the `conv_bn(128, W)`, `GatedResidual(W)`, and line 151 `conv_bn(W, 512)` stem-in. Pass `layer2_width` into `ResNet9.__init__`.
6. Pre-register diagnostics: print `num_epochs` and steady-state `img/s` per cell (already printed). These are the first-class read per the under-anneal gate.

## Expected effect (quantified)
- cell-A (compile-only): +7–15% img/s → ~160–172 epochs. Marginal anneal value near 150 is small but plausibly positive (EXP-007/013 show accuracy STILL CLIMBING when epochs cut below 150). Expected: 0 to +0.1pp. Likely sub-noise → informative null on the throughput lever.
- cell-B (compile + layer2 256→320): if compile's +10% offsets the ~1.25× capacity cost, epochs hold near ~145–150 WITH ~+1.3M params at the proven 8×8 stage that EXP-004 showed is capacity-binding. This is the only cell with a path to clear +0.1pp. Highest upside.

## Risks
- **Compile cost leaking on-budget**: if warmup is misplaced or shape-mismatched, the loop recompiles INSIDE the timed region, catastrophically inflating the first timed steps. Mitigation: warmup at EXACT (512,3,32,32) channels_last bf16; `drop_last=True` guarantees the loop never sees a partial batch.
- **Gains <5%**: on this already-fast net the elementwise share may be small → cell-A ties cell-0. Clean negative result on the throughput lever (still publishable learning).
- **Inductor bf16 numerical drift**: airbench calls it "small float differences"; combined with the ~0.1pp epoch-jitter noise floor, a sub-0.1pp cell-A move is unattributable — hence cell-0 same-session control is mandatory.
- **Eval-path recompile toward 10-min wall**: avoided by design (EMA copy uncompiled). Pre-warmup compiled-`model` evals are few/off-budget. Watch `total_seconds` < 600.
- **cell-B under-anneal**: if compile gain < capacity cost, epochs drop <140 and cell-B under-anneals like EXP-007. num_epochs is the pre-registered gate (≥130 valid).
- **OptimizedModule/optimizer aliasing**: low risk (params shared via `_orig_mod`), but verify in smoke that `optimizer.step()` changes weights the next forward sees.

## Verification approach
Same-session 3-cell run (cell-0 USE_COMPILE=0; cell-A USE_COMPILE=1 LAYER2_WIDTH=256; cell-B USE_COMPILE=1 LAYER2_WIDTH=320), GPU 1, back-to-back one process, under `timeout 600` each. Win requires a cell that beats stored 96.38 + 0.1pp AND beats same-session cell-0 by >0.1pp (the ~0.1pp noise floor makes the stored baseline too weak alone). Pre-register and report num_epochs + img/s per cell; smoke-test the warmup/aliasing invariants before the timed runs.

## Effort
Medium. The compile + warmup is low-effort (~15 lines); cell-B width parametrization and the 3-cell harness add modest plumbing. One experiment loop.

## Sources
- arXiv:2404.00498 "94% on CIFAR-10 in 3.29 Seconds" (airbench): torch.compile ~14% throughput, math-equivalent, one-time compile amortized over many steps.
- `train.py`: budget accounting (lines 268–314, `t_start_training`, per-step `total_training_time`); off-budget whitening (232–238) and local-RNG discipline (line 85); model (140–186); EMA (255–257); eval gating (341–349).
- `prepare.py` Eval: batch 256, `model(inputs)`, `model.eval()`, `inference_mode` (lines 11–47) — last batch 16 (dynamic), avoided by eval'ing uncompiled EMA copy.
- 03-experiment-learnings: under-anneal at fixed TIME budget (EXP-005/007/013); EXP-004 capacity-binding at layer2/8×8; EXP-007 pre-registered milder 256→320 step; ~0.1pp noise floor.
