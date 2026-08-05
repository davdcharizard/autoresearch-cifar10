1. **Plan lines 30-32, Code Changes: zeroing `self.layer2[2].c2[1].weight` makes the new block permanently dead.**  
   The module index is correct: after adding `Residual(256)`, `layer2[2]` is the new residual and `c2[1]` is its second `BatchNorm2d`. But `conv_bn()` is `Conv2d -> BatchNorm2d -> ReLU`, so zero BN gamma makes `c2(...)` exactly zero before a ReLU. PyTorch’s ReLU derivative at exactly zero is zero, so no gradient reaches `c2[1].weight`, `c2[1].bias`, `c2[0]`, or `c1`; weight decay also cannot move zero gamma. The added residual branch stays identity forever. This invalidates the capacity hypothesis and likely only burns throughput.

2. **Plan line 13, Smoke: the identity check verifies the failure mode instead of detecting it.**  
   `torch.allclose(model.layer2[2](h), h)` will pass precisely because the branch is zeroed, but the smoke does not check that the zero-initialized branch can receive gradients. A one-step backward check on `model.layer2[2].c2[1].weight.grad` would expose that it is zero. Without that, Milestone 1 can green-light a non-learning block.

3. **Plan lines 5 and 32: “starts bit-equivalent” is used to justify no LR retune, but the implementation is not a trainable zero-init residual.**  
   Standard zero-init residuals zero the final BN before addition without a ReLU blocking the zero path. This code’s final BN is inside `conv_bn`, which includes ReLU before the residual add. The plan imports the right conceptual trick but applies it to an incompatible local block definition.

4. **Plan lines 58-61, Verification Protocol: scope/genuineness checks are too weak for the stated hard constraints.**  
   `git diff --name-only ...` only proves the changed file is `train.py`; it does not prove the diff is limited to the layer2 block and one init line. `grep "manual_seed(42)" train.py` still passes if another seed-setting line is added later. `grep -c "evaluator.evaluate(" train.py` does not rule out eval circumvention inside `forward()` or other test-set leakage from `train.py`.

5. **Plan lines 46-51, Abort Criteria: no abort or diagnostic covers the most likely concrete failure.**  
   The criteria watch NaNs, wall time, and throughput, but not whether the new residual branch ever becomes active. Given the zero-gamma-plus-ReLU issue, the official run can complete cleanly while testing only “same network with fewer training steps,” not the intended capacity change.

6. **Plan lines 51 vs 13: the smoke-run instructions are internally inconsistent.**  
   Milestone 1 requires a separate correctness smoke, but Abort Criteria says “the official run IS the experiment — no separate smoke run is launched.” That ambiguity matters here because a pre-run gradient smoke is exactly what is needed before spending the official run.
