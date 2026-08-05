# Report EXP-014: torch.compile throughput (off-budget warmup) + compile-funded layer2 256→320

- **Created**: 2026-06-29

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) within the fixed 300s training budget, editing
only `train.py`. Baseline **96.38%** (EXP-008, commit 07c3760); improvement bar **≥96.48** (+0.10pp,
clearly above the ~0.1pp throughput-jitter noise floor).

## Idea & Hypothesis
Chosen (thorough brainstorm, Codex-reviewed pick over standalone mild-capacity and Ghost-BN): **`torch.compile`
the training forward, paying compilation OFF-BUDGET via a warmup before `t_start_training`, and spend the
bought throughput on a mild capacity widen (layer2 256→320)**. Diagnosis: after 7 straight no-improvements,
every lever WITHIN the fixed ~150-epoch budget had saturated (optimizer, eval-TTA, input-aug, reg-scalars,
loss-geometry) and every lever that SPENT epochs under-annealed (EXP-005/007/013). The unifying constraint
appeared to be the **epoch budget itself** — so the meta-lever was to BUY more epochs via faster code.
`torch.compile` is the rare throughput-additive lever (airbench arXiv:2404.00498 got ~14%, math-equivalent;
the off-budget warmup is precedented by the off-budget whitening eigendecomp).

**Hypothesis**: compile yields +7–15% img/s with the compile cost off the 300s budget; cell-A (compile/256)
gains anneal epochs (~150→~165) for ≤+0.1pp (likely sub-noise near the ceiling); **cell-B (compile/320) holds
epochs ≥~140 WITH +1.03M annealing capacity at the proven 8×8 stage → clears 96.48 and beats the same-session
control** by resolving the EXP-007 under-anneal failure. Falsifiable: if compile throughput <5%, or cell-B
under-anneals (epochs <120, best==final), the throughput-funded-capacity thesis is rejected.

## Approach
All changes in `train.py` only, env-toggled (defaults reproduce EXP-008 byte-for-byte). A SEPARATE compiled
handle `train_fwd = torch.compile(model, mode="default")` drives the in-loop forward; `model`/`ema_model`
stay uncompiled for EMA + eval (zero eval recompile). The off-budget warmup (3 fwd+bwd at the exact static
(512,3,32,32) channels_last/bf16 shape, local-RNG dummies, no `optimizer.step`, BN buffers snapshot+restored)
sits before `t_start_training`, gated on `USE_COMPILE or WARMUP` so ALL cells (control included) get the same
off-budget cuDNN-autotune prepay — isolating compile-fusion as the only cell-A−cell-0 difference (plan-review
concern #8). `ResNet9` gained a `layer2_width` param (cell-B: conv_bn(128,320) + GatedResidual(320) +
conv_bn(320,512), ReZero α=0 identity init → no LR retune). First-10-step dt logging on compiled cells guards
against in-loop compile leakage. Three same-session cells, each a SEPARATE `train.py` process under its own
`timeout 600`. A pre-registered smoke (`smoke.py`) verified — for both widths — compile correctness, param
aliasing, BN restore, off-budget invariant, eval-boundary recompile guard, and global-RNG isolation.

## Execution
Smoke PASSED both widths (warmup 24.0s@320 / 12.4s@256 — well under the 120s wall-cap gate; post-warmup
compiled step ~17ms@256 vs uncompiled ~19.7ms, ≈13% faster; no in-loop recompile). **Run 1 was GPU-1
contention-confounded** (infra-error EXP-010 recurrence): a foreign job (PID 1723342) appeared during cell-0
and ramped to 20+ GB / 100% util, unequally slowing the cells (epochs 127/74/64) — discarded. **Run 2 re-ran
all three cells once GPU 1 was idle** (driver [smi] snapshots 0–3% util throughout) → clean and comparable.
All cells exit 0, training_seconds 300.0, total_seconds <600, no NaN, prepare.py byte-unchanged.

## Results
- **Primary metric**: 96.32% (best cell = cell-A) (baseline 96.38, delta **−0.06**; vs **same-session**
  control cell-0 96.29, delta **+0.03**).
- **Table** (best / final / epochs / params) — Run 2, clean:
  - **cell-0** (no-compile/256): 96.29 / 96.24 / **154** / 7,784,627 — strong same-session control, clean band.
  - **cell-A** (compile/256): 96.32 / 96.32 / **173** / 7,784,627 — **+19 epochs (+12%) over cell-0**, accuracy +0.03pp (sub-noise).
  - **cell-B** (compile/320): 96.21 / 96.21 / **143** / 8,817,203 (+1,032,576 = predicted +1.03M) — **−0.08pp vs cell-0**, at a HEALTHY anneal count.
- **Observations**: The throughput mechanism worked exactly as designed — compile bought +12% epochs cleanly
  (off-budget warmup confirmed: warmup_seconds 11.8s, first compiled step 38ms then steady 17–22ms ~27–30k
  img/s, NO in-loop recompile). But neither use of the bought throughput converted to accuracy: (1) cell-A's
  +19 anneal epochs lifted accuracy only +0.03pp → the net is **anneal-saturated** at ~150 epochs (extra
  optimizer updates past the budget's natural epoch count are worth ~0). (2) cell-B's compile-funded 320
  capacity annealed at 143 epochs — a healthy count (without compile, 256→320 lands ~120–130) — yet
  *lost* 0.08pp. Since cell-A proved epochs near 150 are worth ≈0, cell-B's 11-epoch deficit vs cell-0 cannot
  explain the loss → the 256→320 capacity **genuinely does not help even when properly annealed**.
- **Analysis**: hypothesis rejected on both arms. The throughput sub-hypothesis (compile buys epochs) is
  CONFIRMED (+12%); the accuracy sub-hypothesis (epochs/capacity → accuracy) is FALSIFIED. This is the
  decisive result of the experiment: it **disconfirms the "epoch budget is the binding constraint" diagnosis**.
  Given MORE epochs (cell-A) OR more annealed capacity (cell-B), accuracy does not move — the net is at a
  genuine generalization ceiling at ~96.3–96.4 for this architecture/budget, not epoch-bound. It also RESOLVES
  the long-standing EXP-007 ambiguity: 256→320 under-annealed there, but here, properly annealed via compile,
  it still loses → the layer2/8×8 capacity axis is genuinely EXHAUSTED, not epoch-starved.
- **Key Learning**: torch.compile cleanly buys ~12% throughput (154→173 epochs, off-budget warmup works), but
  neither extra anneal epochs (cell-A +0.03pp) nor compile-funded 320 capacity at a healthy 143-epoch anneal
  (cell-B −0.08pp) lifts accuracy — the net is anneal- AND capacity-saturated, NOT epoch-bound.

## Verification
- **Conditions**: NC1 PASS (all cells 300s training, <600s wall, valid metrics, exit 0, no NaN). **NC2 FAIL**
  (best cell 96.32 < 96.48 bar and only +0.03pp over same-session cell-0; no win → no confirmation re-run).
  NC3 PASS (only train.py modified; prepare.py byte-unchanged; seeds intact via LOCAL warmup generator;
  num_params 7,784,627 / 8,817,203; ≤1 eval/epoch).
- **Review Notes**: Run 2 results trustworthy — GPU 1 uncontended throughout (the whole reason for the
  re-run), same-session control reproduced the recipe at a clean 154-epoch draw, throughput diagnostic
  internally consistent (cell-A epochs 173 > cell-0 154 confirms the compile gain), no instability, anti-
  bookkeeping clean (best == max per-epoch). Run 1 correctly discarded as infra-confounded.
- **Verdict**: no-improvement
- **Verdict Basis**: valid, clean result; NC2 (result-quality gate) failed — no cell cleared the bar.

## Unexplored Avenues
- **Spend the freed ~12% throughput on a DIFFERENT axis than capacity/epochs.** Compile is a real, banked
  ~12% throughput surplus that future experiments can use for free (it's now a proven, math-equivalent lever).
  Capacity (cell-B) and raw epochs (cell-A) both failed, but the surplus could fund a per-step-COSTLIER
  *regularizer/optimizer* that previously under-annealed — most notably a **near-free SAM variant or even
  plain tail-SAM** (EXP-013 lost only because its 2× cost removed ~26 epochs; compile buys ~18 of those back).
  Honest caveat: EXP-013 showed zero positive SAM signal even where it ran, so EV is low.
- **Different capacity LOCATION/shape** (not layer2 width): the 8×8 width axis is now exhausted (EXP-004 add
  helped, EXP-007/014 widen does not), but capacity at layer1 (16×16) or a wider stem, or a different block
  topology, is untried — though the ceiling evidence argues against any same-architecture capacity move.
- **max-autotune compile mode**: might extract a few more % throughput than `default`; irrelevant to accuracy
  given epochs don't help, so not worth a loop on its own.

## Next Steps
1. **Accept the generalization ceiling for this architecture; pivot to a DIFFERENT base architecture**
   (confidence: medium-high). EIGHT straight no-improvements (EXP-006→014) now span optimizer, eval-TTA,
   input-aug, reg-scalars, loss-geometry, raw epochs, AND properly-annealed capacity — the last two newly
   closed by EXP-014. The evidence for a real ~96.3–96.5 ceiling on this whitened ResNet-9 at 300s is now
   strong (it is NOT epoch-bound). The highest-EV remaining move is a genuinely different backbone (e.g. a
   wider/deeper airbench-style net, or a pre-activation/anti-aliased variant) that compile's banked throughput
   can help fit in-budget.
2. **A throughput-free regularization mechanism not yet tried** (confidence: low-medium) — Ghost BatchNorm
   (idea-03, shelved) is the one major DavidNet recipe component never adopted; distinct (BN-statistic noise).
   Honest: the regularization-bound saturation argues it likely ties, but it is the last untried throughput-
   free axis before conceding the architecture.
3. **Bank torch.compile into the base recipe regardless** (confidence: high it's free) — it's math-equivalent
   and +12% throughput; even though it didn't help alone, it should ride along with any future experiment that
   needs the headroom (e.g. funding a different architecture's per-step cost). NOT a standalone win.

## Exit Action Results
- None defined (autopilot goal) — section intentionally empty.
