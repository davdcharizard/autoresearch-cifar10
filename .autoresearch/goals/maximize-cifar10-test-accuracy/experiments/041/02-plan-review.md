# Plan Review EXP-041

Offline fallback review found no blocking scope or single-score flaw, but the
following execution-level corrections are required:

1. **Verification Procedure 5-6 cannot numerically call the stated
   "production objective" as planned.** Both loss branches are inline inside
   `main()`, while the preflight is forbidden from constructing the real
   loader/evaluator. A locally reconstructed loss can pass even if scored
   `main()` uses the wrong logits, targets, coefficient, or a second mixup
   draw. Treat the numeric fixture as a formula oracle only, and separately
   AST/source-audit the actual two branches: one `mixup_batch` call; one
   opt-in dual forward on its returned `mixed_inputs`; the same `mix`,
   `targets_a`, and `targets_b` in both path losses; then the exact nested
   expression `(1-s)*main_loss + s*direct_loss`. The hard branch must use one
   opt-in dual forward on `inputs` and the same hard `targets` twice.

2. **Verification Procedure 4's hook wording does not by itself prove
   main/direct ordering.** Record each `fc` hook input and require call 1 to be
   bitwise the independently reconstructed `z + 0.1*h(z)` and call 2 to be
   bitwise raw `z`; invocation count plus output equality alone can miss a
   swapped implementation when only the returned tuple is relabeled. Run the
   default identity oracle against the independently loaded `git show`
   accepted module, not against a candidate-derived reconstruction.

3. **Verification Procedure 6 must keep loss-gradient and optimizer-gradient
   semantics separate.** Obtain main-only, direct-only, and combined gradients
   from three cloned models at the same parameter and BN state for each of
   early mixup and hard labels. Check every pooled-head tensor, with
   direct-only gradients specifically `None` or exactly zero, before comparing
   `g_combined` to the FP32-tolerant weighted sums. Do not include coupled
   decay in these identities; add it only in the update oracle. Print the
   declared absolute/relative tolerances before assertions.

4. **Verification Procedure 7 is ambiguous between "complete" updates and
   "representative" tensors and does not state PyTorch SGD's exact rule.**
   Compare every parameter and every resulting momentum buffer, in both
   regimes, using `d = grad + wd*p0`; fresh `b1 = d`, preseeded
   `b1 = 0.9*b0 + d`; and Nesterov `p1 = p0 - lr*(d + 0.9*b1)`. Use
   `wd=5e-4` only for rank-at-least-two tensors and zero for classifier bias
   and BN parameters. This prevents an oracle that silently applies decay
   after momentum, initializes the fresh buffer incorrectly, or checks only a
   convenient subset.

5. **Milestone 3 and Verification Procedure 8 disagree on timing sample
   count.** The chosen-idea contract requires at least four retained windows
   per implementation per regime; "four paired windows" and `A/C/C/A` can be
   read as only two retained windows per implementation. Require two complete
   `A/C/C/A` cycles (four accepted and four candidate windows) after at least
   20 warmups for each regime. Define `a_mix`, `c_mix`, `a_hard`, and `c_hard`
   as median seconds per complete step, restore equivalent model/optimizer
   fixtures at each paired window, reset peak-memory accounting, and compute
   retention only as the stated weighted sum of reciprocal medians.

6. **The source-scope audit should allow exactly the necessary forward
   signature change as well as its return block.** The current phrase
   "production scope only in the forward return" could reject the required
   default-false argument or, if implemented loosely, permit unrelated edits
   anywhere in `forward()`. Compare against `git show a7c42dc:train.py` and
   whitelist only the signature, pooled/refined/main/direct return block, and
   the two inline loss branches; all other production text must remain
   accepted.

The timing equation and threshold, exact nested 90/10 arithmetic, default
inference contract, Nesterov/decay intent, fixed temporal controls, and sole
scored command are otherwise coherent once these verifier ambiguities are
closed.
