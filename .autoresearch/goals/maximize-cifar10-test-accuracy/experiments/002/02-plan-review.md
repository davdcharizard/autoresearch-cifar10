1. **High - Verification Protocol §4 (`02-plan.md:106`)**: The no-reward-hack check is brittle. Literal greps for `train=False` and `evaluator.loader` would miss positional/aliased test-set access, while the `CIFAR10(` check is ambiguous because current `train.py:137-139` splits `CIFAR10(` and `train=True` across lines. This could pass test-set leakage while appearing compliant.

2. **High - Code Changes §6 (`02-plan.md:65-76`)**: After `ema_started`, the plan evaluates only `ema_model`, never the raw model. That contradicts the “raw trajectory is preserved as a floor” framing in the brainstorm/proposal; `best_acc` can fall below EXP-001 if EMA lags or BN buffers mismatch, even if the raw model would have matched baseline.

3. **High - Abort Criteria (`02-plan.md:94,96`)**: The fallback actions drift outside the reviewed config. “Drop EMA,” “disable TTA,” or “reduce batch” are different experiments from “EMA + flip-TTA, everything else unchanged,” and using them after seeing signals risks post-hoc variant selection.

4. **Medium - Smoke Test (`02-plan.md:16`)**: The smoke test claims to confirm `evaluate(ema_model, device)` reachability but never calls `evaluator.evaluate`. It misses the actual frozen path in `prepare.py:33-47`, including Eval’s DataLoader tensors, Eval’s `.eval()` call, and CE-loss execution.

5. **Medium - BN Handling (`02-plan.md:48-55,75-76`)**: The plan treats `use_buffers=True` as solving BN consistency, but EMA-averaged BN buffers are not recomputed BN stats for the averaged weights. If accuracy regresses, this is a concrete failure mode the plan currently folds into generic “EMA decay may lag.”

6. **Medium - Timing Verification (`02-plan.md:103`; `train.py:27,120,251-263`)**: Printed `total_seconds` starts inside `main`, after top-level `evaluator = Eval()` has already constructed/downloaded the test set. The `timeout 600` wrapper is the real wall guard; `total_seconds < 600` can under-report process wall time.

7. **Low - TTA Scope Verification (`02-plan.md:41-44,106`)**: “Exactly one `evaluator.evaluate` call” does not bound eval-time compute inside `forward`. The planned code uses one flip, but verification would also pass a many-view TTA loop hidden in `forward` if it stayed under wall timeout.
