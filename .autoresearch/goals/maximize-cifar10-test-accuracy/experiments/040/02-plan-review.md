# Plan Review EXP-040

Offline fallback review found no blocking issue and required these clarifications:

1. Only the instantaneous coupled-decay gradient `weight_decay * W` is purely radial; historical Nesterov buffers need not align with current rows and can change directions/ratios.
2. Verify the analytic raw-weight derivative using a fixed upstream `q` independent of `W`, such as `(W_eff*q).sum()`; CE-derived `q` must be detached before use in that oracle.
3. Restrict row-rescaling equivariance to a positive multiplier because a negative multiplier flips direction.
4. Load accepted code independently with exact `git show a7c42dc:train.py`; keep textual source-diff reconstruction as a separate scope audit.
5. Treat nonzero rows in fixed fixtures as sampled evidence, with runtime finite-loss handling for the scored trajectory.
6. State separate semantic and timing timeouts rather than an inconsistent combined preflight duration.

The formula, invariants, optimizer preservation, retention equation, score threshold, source scope, and narrow closure are otherwise sound.
