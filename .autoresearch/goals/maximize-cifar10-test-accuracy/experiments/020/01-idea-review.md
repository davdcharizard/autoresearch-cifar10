Prioritized Feedback

1. Idea-01 is the cleanest EXP-020, but stop overselling the MosaicML number. The brainstorm cites a ~0.5% cosine-vs-cyclic gap, but this is not this exact DavidNet, augmentation stack, 150-epoch budget, or EMA/TTA setup. Fix: pre-register only `SCHEDULE=cos` with unchanged `PCT_START=0.15` as the verdict cell; treat shorter-warmup `cB` as EXP-021 material if it wins.

2. Idea-01 needs an LR-trace sanity check before the real run. The current schedule is exactly linear post-warmup in `train.py:286-290`, so `SCHEDULE=tri` must reproduce it bit-for-bit. Also correct the proposal’s “cosine decays faster initially” wording: cosine holds LR higher early after warmup, crosses linear near mid-decay, then spends more time at very low LR. Fix: print sampled `progress, lr_tri, lr_cos` plus fraction of steps below LR thresholds.

3. Idea-02’s evidence is mostly recipe correlation, not an activation ablation. `hlb`/`airbench` using GELU does not prove GELU caused accuracy; those recipes differ in schedule, optimizer, architecture, and training regime. The proposal admits this risk. Fix: if run later, log `num_epochs`, ep25, activation/gradient stats, and use `nn.GELU(approximate="tanh")` or explicitly verify exact GELU does not cut epochs below the ≥135 gate.

4. Idea-02 has an init/optimization hidden assumption. `train.py:157-160` uses Kaiming init with `nonlinearity="relu"`. BN reduces the mismatch, but replacing every `conv_bn` activation at `train.py:101-106` changes branch dynamics globally. Fix: keep ReLU init for the first clean single-variable test, but do not interpret an early-convergence drop as a definitive activation-axis failure without a follow-up LR/init diagnostic.

5. Idea-03 has the most fragile implementation. A raw learnable `logit_scale = nn.Parameter(torch.tensor(INIT_TAU))` can become non-positive and will receive weight decay under the current optimizer grouping. That can create a false negative unrelated to cosine geometry. Fix: parameterize `log_tau`, use `tau = exp(log_tau)` with a sane clamp/range, exclude it from weight decay, and log tau every epoch.

6. Idea-03 also needs scale matching before claiming a geometry result. The current head is `self.fc(x) * self.scale_out` at `train.py:153,178`; replacing it with normalized features and `tau≈14` changes both geometry and initial logit magnitude. Fix: run a smoke comparing baseline vs cosine-head initial logit std/range on one batch; choose `INIT_TAU` to match baseline logits or make mismatch explicit.

7. Across all ideas, same-session control and confirmation are mandatory. The goal’s noise floor is ~0.1pp, and EXP-019 showed a +0.28pp same-session lead collapsing to +0.02pp on confirmation. Fix: require `cA ≥ 96.48`, `cA - c0 > 0.1pp`, and a confirmation rerun before declaring a win.

Scored Verdict

| Idea | Evidence / reasoning | Potential impact |
|---|---:|---:|
| Idea-01: cosine one-cycle decay | 8/10. Best-supported: directly changes the untried schedule shape, is flagged by EXP-012, targets the low-LR tail, and is truly throughput-free. | 7/10. Most plausible cheap path to +0.1pp or more; optimistic external upside may shrink, but it can clear the bar without paying epoch cost. |
| Idea-02: GELU activation | 5/10. Mechanism is reasonable, but evidence is mostly borrowed from confounded fast-CIFAR recipes rather than direct ablations. | 4/10. Cheap and worth a later probe, but likely sub-noise on a BN-heavy saturated ResNet-9. |
| Idea-03: cosine classifier | 4.5/10. Real mechanism, but the strongest literature fit is face/open-set/angular-margin classification, not balanced CIFAR-10; implementation scale sensitivity is high. | 4/10. Could help if magnitude coupling is genuinely harmful, but the upside is modest and the risk of a bad tau/LS interaction is larger than the proposal admits. |

Pick for EXP-020: Idea-01, cosine one-cycle decay.

It wins because it has the strongest combination of internal history, external schedule evidence, low implementation risk, and direct relevance to the diagnosed limiter. Idea-02 is a plausible representation probe but its evidence is mostly correlational. Idea-03 is interesting but too sensitive to temperature/scale details for a first move after 14 nulls. Run cosine as the single primary cell, keep the same warmup, make `tri` an exact regression control, and treat any diagnostic variant as non-verdict.
