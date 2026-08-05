# Adversarial Review - EXP-022 Candidate Ideas

## Prioritized Feedback

1. **Sparse SAM under-spends the measured cost envelope.** One-in-four use during the final 10% retains about 97% exposure but may be too weak to change the terminal basin. Use every other final-window step; EXP-021 timings still project about 94-95% whole-run retention, above the 90% gate.
2. **RandAugment compounds the accepted early regularizer.** Applying it throughout training repeats the additive-regularization pattern rejected by CutMix, stronger mixup, and dropout. A future test should use it only in the hard-label tail and require a warm loader-throughput gate.
3. **Alpha 0.1 has a low ceiling and adverse local evidence.** Its endpoint-heavy Beta law is closer to less mixup, while the shorter mixup window already regressed.
4. **SAM requires hard semantic and timing preflights.** Exact parameter restoration, a single persistent BatchNorm update, finite gradients, and the measured >=90% whole-run exposure projection must pass before scoring.

## Scored Verdict

- **Sparse Final-Window SAM**: evidence 7/10, impact 7/10. It targets terminal geometry and quantitatively repairs EXP-021's cost failure, but needs the denser every-other cadence.
- **Mild RandAugment**: evidence 6/10, impact 7/10. Strong CIFAR literature support, but in-window stacking conflicts with repeated local regularization failures.
- **Weaker Alpha-0.1 Mixup**: evidence 5/10, impact 3/10. Controlled but locally predicted to regress and unlikely to clear the acceptance margin.

## Pick

**Sparse Final-Window SAM, refined to every other final-window step.** It is the only candidate that directly targets solution geometry without adding early regularization, and EXP-021 supplies measured timing and validated semantics. The denser cadence spends available exposure headroom while retaining a preregistered feasibility margin.
