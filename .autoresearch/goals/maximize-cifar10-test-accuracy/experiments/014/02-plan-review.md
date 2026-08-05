1. **[Plan §Abort Criteria lines 99-101 / NC1 lines 115-117] In-loop compile leakage can pass NC1.**  
   `train.py:280-314` charges any missed compile/recompile into that step’s `dt`, but the loop will still run until `training_seconds≈300`. A leaked 20-60s compile will mostly show up as fewer steps/epochs, not `training_seconds≪300`. Current logs print only every 50 steps, so a first-step compile can be invisible. Fix: log first 5 step `dt/img/s` for compiled cells and fail on step-level compile spikes plus abnormal `num_steps`, not just `training_seconds`.

2. **[Plan §Milestone 2 lines 25-27] Smoke is underspecified and may not exercise the real path.**  
   The plan says “inline smoke” but provides no exact command/code. A toy smoke can miss the actual inserted warmup, dataloader-shaped input path, EMA update site, and `Eval.evaluate` mode transition. Fix: pre-register exact smoke code using the real `ResNet9`, real optimizer/criterion, actual warmup block, one raw eval boundary under `torch.inference_mode()`, then two timed train steps.

3. **[Plan §Milestone 2 line 26 / §Configuration line 87] Headline width may not be smoked.**  
   The smoke description does not require `LAYER2_WIDTH=320`, but cell-B is the headline and compiles a different graph with different conv shapes. A 256-only smoke does not validate 320 compile latency, BN restore, aliasing, or wall risk. Fix: smoke the exact headline config (`USE_COMPILE=1 LAYER2_WIDTH=320`) and at least a 256 compiled control if cell-A remains part of the protocol.

4. **[Plan §Execution Environment lines 91-94] Pre-run smoke can populate Inductor cache and hide wall-clock risk.**  
   If the smoke compiles the same graph before official cells, PyTorch/Inductor cache reuse may make the official process pass `timeout 600` only because compilation happened outside the run. The plan explicitly says compile warmup is on the wall clock. Fix: run smoke with an isolated throwaway `TORCHINDUCTOR_CACHE_DIR`, and run each official cell with a fresh cache or explicitly record cache reuse as invalid for wall-cap validation.

5. **[Plan §Execution Environment line 93] 600s wall estimate is not hardened.**  
   Existing cells are ~447s wall before compile; the plan assumes compile warmup is only 30-90s. A cold training fwd+bwd compile can be multi-minute, pushing a compiled cell over 600s. Fix: make cold compile warmup duration a smoke output and abort before official cells if projected `startup + train + eval` approaches the cap.

6. **[Plan §Verification Protocol line 119] Confirmation rerun gate is too weak.**  
   Initial win requires `> c0 + 0.10pp`, but confirmation only requires `best ≥96.48` again. That can accept a non-reproduced same-session advantage after selecting the best of two compiled cells. Fix: confirmation must either rerun a paired control in the same session and beat it by >0.10pp, or require the winning cell to reproduce both the absolute bar and the original over-control margin.

7. **[Plan §Abort Criteria line 101 / §Verification line 111] GPU contention control is not sufficient for sequential cells.**  
   One pre-run `nvidia-smi` does not protect a 22-25 minute sequential run. Contention can appear after cell-0 or affect only one cell, making same-session ranks invalid. Fix: check/log GPU occupancy before every cell and poll during cells, or rerun cell-0 adjacent to any winning confirmation.

8. **[Plan §Code Changes lines 54-75] Compile warmup also moves cuDNN/autotune/cache warmup off-budget, but cell-0 does not get an equivalent control.**  
   With `torch.backends.cudnn.benchmark=True`, the dummy fwd/bwd can prepay non-compile kernel selection and allocator warmup. Then cell-A vs cell-0 is not pure compile throughput. Fix: either give the no-compile control an equivalent off-budget fwd/bwd warmup with BN restore, or explicitly separate generic warmup gain from compile gain in diagnostics.

9. **[Plan §Log output line 94 / §NC3 line 121] Ignored driver script is an integrity blind spot.**  
   `.autoresearch/` is gitignored, so `git status --porcelain` will not reveal `experiments/014/run_cells.sh`. A hidden driver can add retries or selection logic while NC3 still says “only train.py modified.” Fix: do not use a driver script, or record its full contents and exact launched commands in the execution artifact; assert only c0/cA/cB plus mandatory confirmation were run.
