# Report EXP-010: SiLU (Swish) activation in place of ReLU
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-010.md
- **Plan**: plans/plan-010.md
- **Log**: logs/exp-log-010.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%) under a fixed 300s budget on one H20, editing only `train.py`. Higher is
better. Baseline **96.00%** (EXP-003); success bar ≥ **96.10%**.

## Idea & Hypothesis
Chosen idea: swap ReLU→SiLU (Swish, x·sigmoid(x)) at all three activation sites of the k=4 recipe, with the
validated `torch.compile(reduce-overhead)` enabler to keep the test epoch-fair. Rationale: the model is *converged*
at ~77 epochs (EXP-007 showed more epochs don't help) and sits at the capacity sweet spot, so test accuracy is
generalization-bound at fixed capacity — the only live levers improve generalization of the same-size model. SiLU
is the one architectural axis never tried (smooth, non-monotonic, EfficientNet default), nearly free, and orthogonal
to the six exhausted axes. Hypothesis: a small generalization gain lifting `best_test_acc` to ~96.1–96.3%.

## Approach
`train.py`-only edits: `F.relu`→`F.silu` at the 3 sites (BasicBlock post-bn1 + post-residual, ResNet stem);
`compiled_model = torch.compile(model, mode="reduce-overhead")` with the training forward routed through it; eval
left eager. Everything else byte-identical to EXP-003 (k=4, Cutout(16), PEAK_LR 0.2, WD 1e-4, label smoothing,
batch 128, bf16, channels_last, Nesterov, cosine, seed 42). SiLU is parameter-free, so num_params is unchanged — a
built-in sanity check that only the nonlinearity changed.

## Execution
One run, no retries/errors, clean compile (no graph breaks), exit 0. Throughput matched compiled-k4 almost exactly:
steady-state **dt = 9ms/step (~14,785 img/s)** vs EXP-007's 8ms — SiLU's extra elementwise sigmoid was negligible
and the compile enabler absorbed it. The run fit **85 epochs** (vs k=4's 77, compiled-k4's 89) — a *fair, fully-
converged* test, well past the ~77 convergence point. Completed in 400.3s total, peak VRAM 548.7 MB, params
4,299,866 (unchanged).

## Results
- **Primary metric**: **95.73%** (baseline 96.00, delta **−0.27 pp**, −0.28%) — below the +0.1 bar and baseline.
- **Observations**: final_test_loss **0.2136** ≈ compiled-k4's 0.208 ≈ EXP-003's 0.204 — SiLU did not reduce the
  loss. Late evals plateaued 95.63–95.73 (ep 83–85). SiLU-k4 (95.73) ≈ compiled-k4 (95.92, EXP-007) within the
  ~0.2pp noise band — i.e. SiLU added no measurable accuracy (if anything marginally below ReLU).
- **Analysis**: A clean negative — the feared epoch starvation did not occur (85 epochs, dt 9ms), so SiLU got a
  fully-converged fair shot and still did nothing. With BatchNorm already smoothing the optimization landscape, the
  marginal benefit a smoother activation provides is evidently absent for this compact k=4 CIFAR ResNet. This is
  consistent with the established picture: the model is generalization-bound at fixed capacity, and a nonlinearity
  swap doesn't change the generalization ceiling here. Adds a **seventh** exhausted axis (activation/nonlinearity)
  to width, regularization, weight-averaging, training-length, channel-attention, and compiled-capacity-scaling.
  96.0% is an increasingly robust plateau.
- **Key Learning**: ReLU→SiLU adds no accuracy to the converged k=4 WideResNet on CIFAR-10 at this budget (fair
  85-epoch run, 95.73 ≈ compiled-k4 95.92, loss 0.214 unchanged) — the nonlinearity axis is non-binding here.

## Verification
- **Conditions**: Cond 1 (clean completion in budget) PASS; Cond 2 (≥96.10) **FAIL** (95.73); Cond 3 skipped.
- **Review Notes**: Trustworthy — clean single run, frozen eval, seed 42, eval once/epoch (85=85), num_params
  UNCHANGED (4,299,866) confirms the parameter-free swap (no accidental architecture change), compile is
  execution-only with EXP-007-established null standalone accuracy effect → the null is attributable to SiLU
  itself, not to under-training (85 epochs) or compile. No reward-hacking surface. The −0.27pp is within noise.
- **Verdict**: no-improvement
- **Verdict Basis**: valid, trustworthy, well-trained run; primary metric below the +0.1 bar (cond 2 failed).

## Unexplored Avenues
- **Mixup/CutMix** (brainstorm-010 idea 3) — a complementary regularizer (interpolation, not occlusion). Different
  mechanism than the exhausted axes, but typically needs more epochs than the converged 77-budget allows → likely a
  soft regression. Low-medium priority.
- **Per-channel input std normalization** (brainstorm-010 idea 2) — fix `std=(1,1,1)`. Near-free but the immediate
  BN largely absorbs it → expected near-neutral. Low priority, but cheap enough to be worth one clean test.
- **LR-schedule / optimizer micro-tuning on k=4** — peak-LR and warmup-fraction were never swept (only WD in
  EXP-005). A genuinely-untried recipe knob on the converged sweet spot. Low-medium priority.
- The activation/nonlinearity idea is now exhausted (Mish/GELU are near-identical smooth activations, unlikely to
  differ from SiLU's null).

## Next Steps
1. **Per-channel input std normalization** (fix `std=(1,1,1)` → CIFAR std) — *low confidence*; near-free, textbook-
   correct, untried, but BN likely absorbs it. Cheapest remaining clean probe. *Best next experiment* (low cost,
   clean attribution).
2. **LR-schedule micro-tuning** (e.g. peak-LR 0.2→0.3, or warmup fraction) — *low-medium confidence*; an untried
   optimization knob, but a single run only tests one point of a sweep.
3. **Accept 96.0% as a hard plateau** — *strategic*; SEVEN axes now exhausted. Remaining moves are noise-scale;
   convergence is essentially reached. After 1–2 more cheap probes (std-norm, one LR point), declaring the plateau
   is the honest call.

## Exit Action Results
- None defined for this goal — skipped.
