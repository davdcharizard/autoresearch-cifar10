1. **Severe: metric verification can still be gamed**  
   `02-plan.md:104-106` trusts the summary `best_test_acc:` from editable `train.py`, but does not cross-check it against the per-epoch `eval ep ... test_acc/best` lines produced immediately after `Eval.evaluate` (`train.py:349-356`, `train.py:372-382`). A train.py bug or reward-hack could inflate only the summary and pass.  
   **Fix:** require exactly one summary line, `eval_count == num_epochs`, and `summary best_test_acc == max(per-epoch test_acc/best)` before any win is accepted. Run integrity checks regardless of whether accuracy passes.

2. **High: under-anneal gate is stated but not enforced**  
   The hypothesis and Smoke E use `>=135` epochs (`02-plan.md:6`, `02-plan.md:21`), but Abort Criteria only discard `<~130` (`02-plan.md:94`) and Verification makes `135-154` only informational (`02-plan.md:108`). A 130-134 epoch blur cell could still be accepted.  
   **Fix:** make actual `num_epochs >=135` a hard pass condition for all official and confirmation cells; otherwise restrict layers/re-run the full same-session set.

3. **High: throughput smoke is underspecified for the real bottleneck**  
   Smoke E says “~200 steps img/s” (`02-plan.md:21`) but does not require full training steps with autocast, loss, backward, optimizer, synchronize, and EMA conditions. Forward-only timing would miss BlurPool backward and `F.pad` overhead.  
   **Fix:** define Smoke E as full train-step timing matching `train.py`’s loop.

4. **High: GPU contention detection only samples before runs**  
   The plan acknowledges mid-session contention risk (`02-plan.md:93`) but only logs `nvidia-smi` before each cell (`02-plan.md:24`, `02-plan.md:108`). A foreign job appearing mid-cell can bias epochs and same-session deltas.  
   **Fix:** run periodic/background `nvidia-smi` sampling during each cell and discard/re-run the full set on mid-run contention or unexplained step-time drift.

5. **Medium: confirmation gate is still hairline for the known noise floor**  
   Verification takes `M = max(cA, cB)` and accepts `M > c0 + 0.10` with one confirmation pair (`02-plan.md:105`). With ~0.1-0.2pp noise and prior low-control artifacts, replicated 0.11pp margins are still fragile.  
   **Fix:** pre-register a borderline rule: e.g. if either paired delta is `<0.15pp`, run a third c0/winner pair or require a stronger mean paired delta.

6. **Medium: BlurPool smokes miss eval/TTA/EMA coverage**  
   The proposed code’s shape math should produce 16/8/4, and fp32 buffer + autocast should work, but the smokes only cover train/autocast backward (`02-plan.md:18-20`). They do not test native-fp32 eval through `prepare.Eval`, flip-TTA, or EMA buffer invariance despite claims at `02-plan.md:75-78`.  
   **Fix:** add eval-mode/no-autocast smoke with `tta=True`, finite logits for original+flipped inputs, and post-update `ema_model` blur buffers equal to raw buffers within tolerance.

7. **Low: log paths are inconsistent**  
   Commands write `run_c0.log` in project root (`02-plan.md:25-27`), while Execution Environment says logs are under `experiments/018/` (`02-plan.md:89`).  
   **Fix:** choose one path and use it consistently in redirects and grep commands.

**Overall verdict:** not execution-sound as written. The core BlurPool implementation plan has no obvious blocking shape/dtype/padding bug, but the verification and under-anneal controls need tightening before an EXP-018 result should be trusted.
