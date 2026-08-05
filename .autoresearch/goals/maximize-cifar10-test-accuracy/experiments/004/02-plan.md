# Plan EXP-004: Identity-initialized layer2 residual block (capacity probe)
- **Created**: 2026-06-28

## Summary & Baseline
Add one **ReZero-gated** residual block to `layer2` (the only stage without a residual): `x + α·c2(c1(x))` with a learnable scalar `α` initialized to **0**, so the deeper net starts **exact identity → bit-equivalent to the proven EXP-003 net**, and `α` (which receives a nonzero gradient from the branch output) ramps the new capacity in gradually as training proceeds. `PEAK_LR` stays at **0.4** (no LR retune → clean single-variable capacity test, no depth+LR confound). Whitening + EMA + flip-TTA unchanged. Current baseline (from `04-results.tsv`): **95.87%**; improvement bar **≥95.97%** (+0.1pp).

> **Design correction (plan-review #1–3, fatal bug avoided):** the originally-planned identity-init via zeroing the new block's final BatchNorm γ is **broken** for this codebase, because `conv_bn` = Conv→BN→**ReLU**: a zeroed final BN gives `c2(c1(x))=ReLU(0)=0`, and ReLU's derivative at 0 is 0, so **no gradient ever reaches the block** (γ, β, both convs) and it stays identity *forever* — testing "same net, fewer epochs", not capacity. The standard zero-init-residual trick requires the zeroed BN to be the last op before the add (no intervening ReLU). **ReZero** (a learnable scalar gate, `α`=0 at init) achieves the same "start-as-identity" property while keeping a live gradient path (∂L/∂α = ⟨grad_out, branch_out⟩ ≠ 0), so the block genuinely learns. This is the corrected approach.

## Milestones

### Milestone 1: Code change implemented + correctness/gradient smoke
- [ ] Add a `GatedResidual(nn.Module)` class next to `Residual` (`train.py`, after line 116): `c1=conv_bn(c,c)`, `c2=conv_bn(c,c)`, `self.alpha=nn.Parameter(torch.zeros(1))`; `forward: return x + self.alpha * self.c2(self.c1(x))`.
- [ ] Edit `train.py:129` — add the gated block to `self.layer2`: `nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2), GatedResidual(256))`. (`α` is a bare `nn.Parameter`, untouched by `self.apply(self._weights_init)` which only kaiming-inits Conv2d/Linear; the c1/c2 convs ARE kaiming-init'd. No extra init line needed.)
- [ ] `uv run python -c "import py_compile; py_compile.compile('train.py', doraise=True)"` → exits 0.
- [ ] Smoke (`PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 uv run python` one-shot, in-process, NO training run): build `ResNet9()`, assert:
  - (a) **identity at init** — for `h=torch.randn(2,256,8,8)`, `torch.allclose(model.layer2[2](h), h, atol=1e-6)` (α=0 ⇒ branch contributes 0).
  - (b) **gradient flows to the gate (the bug the design correction guards against)** — run one forward+backward through the full net on `torch.randn(2,3,32,32)` with a dummy loss; assert `model.layer2[2].alpha.grad is not None and model.layer2[2].alpha.grad.abs().item() > 0`. (Confirms the new block is *trainable*, not dead.)
  - (c) full `_forward_once` on `torch.randn(2,3,32,32)` returns shape `[2,10]`, finite.
  - (d) the feature map entering `self.pool` is `512×4×4` (spatial chain intact).
  - (e) `learnable` params increased by the two `conv_bn(256,256)` + the `α` scalar = 1,180,672 + 1 over EXP-003's 6,602,496 → **7,783,169**.
  - (f) `whiten.weight.requires_grad == False`.

### Milestone 2: Official run + early-trajectory gate
- [ ] Launch the official run (background, GPU 1) capturing to `run.log`.
- [ ] Within the first ~30s, read the `step ... img/s` print: confirm projected epochs ≥ ~130 (img/s not catastrophically below EXP-003's ~29.3k → ~174 ep). The time-based schedule guarantees the anneal completes regardless, but <130 epochs raises the under-annealing risk — if projected epochs < ~110, note it as a red flag for analysis (do not auto-abort solely on this; the run is cheap).
- [ ] Early-convergence check vs EXP-003 (identity init should **not** disrupt early epochs): `eval ep 1` test_acc within noise of EXP-003's 60.19% (alarm if < ~55%), `ep 10` near 85.5% (alarm if < ~80%). A large early deficit would mean the identity-init assumption is violated → flag for analysis.
- [ ] Confirm no NaN/divergence and `wall` stays well under 600s.

### Milestone 3: Completion + verification
- [ ] Run completes with a full summary block (`best_test_acc:` present), `total_seconds` < 600.
- [ ] Run the Verification Protocol below; record results in `03-execute.md`.

## Code Changes
- **`train.py` (new class, after `Residual` at line 116)** — add:
  ```python
  class GatedResidual(nn.Module):
      def __init__(self, c):
          super().__init__()
          self.c1 = conv_bn(c, c)
          self.c2 = conv_bn(c, c)
          self.alpha = nn.Parameter(torch.zeros(1))  # ReZero gate: identity at init, learns capacity in

      def forward(self, x):
          return x + self.alpha * self.c2(self.c1(x))
  ```
  *Why*: ReZero (Bachlechner et al. 2020, arXiv:2003.04887). `α=0` ⇒ block is exact identity at init, but ∂L/∂α = ⟨grad_out, c2(c1(x))⟩ is nonzero, so α moves off zero and the branch (and its convs, once α≠0) trains — a *live* gradient path, unlike the broken BN-zero variant (which the post-BN ReLU kills). Lets us start bit-equivalent to the proven net AND keep `PEAK_LR=0.4` (the gradual ramp removes the stability rationale for an LR cut).
- **`train.py` (line 129)** — change
  `self.layer2 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2))`
  → `self.layer2 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2), GatedResidual(256))`.
  *Why*: adds 2 learnable conv layers (8→10 convs, matching airbench96's conv count) at 8×8 resolution — the never-probed capacity lever. Channel- and spatial-preserving (256→256, 8×8→8×8), so `layer3`/`pool`/`fc` and the whiten front-end are untouched. Risk: more params, fewer epochs (under-annealing — see Abort).

No other edits. `Residual` (layer1/layer3) is **unchanged** — only layer2 gains the new gated block, so those proven blocks keep their kaiming init (cleanest attribution). Optimizer (`requires_grad`-filtered, excludes frozen whiten; `α` is included and trains), schedule, EMA construction (`AveragedModel` deep-copies the new architecture incl. `α`), flip-TTA, `_forward_once`/`forward` TTA logic, Cutout, batch, seed 42 — all byte-identical.

## Configuration Changes
- None. `PEAK_LR=0.4`, `PCT_START=0.15`, `EMA_*`, `TTA_START_FRAC=0.8`, `WEIGHT_DECAY=5e-4`, `LABEL_SMOOTHING=0.2`, `BATCH_SIZE=512` all unchanged. (Deliberate: the ReZero gate's gradual capacity ramp removes the stability rationale for the airbench 0.78× LR cut, so this is a single-variable capacity test at the validated LR. `α` gets WD 5e-4 — negligible pull on a scalar that's gradient-driven off zero.)

## Execution Environment
- **Method**: local, `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` (GPU 1 — GPU 0 in use, hard constraint).
- **Resources**: single NVIDIA H20, GPU 1. Expected peak VRAM ~1.8–2.0 GB (the new 8×8 block's activations are tiny; non-binding vs 98 GB).
- **Estimated runtime**: ~7.5 min wall (300s training + eval + startup), well under the 600s kill. Expected ~145–155 epochs (from 174; ~12–18% FLOP increase).
- **Log output**: `run.log` in project root (redirected; do not flood context — grep it). Launch in background, monitor early epochs, block on completion.
- **Tool skill**: none (local run).

## Abort Criteria
- **NaN/Inf or diverging loss** (e.g. `loss` print becomes `nan`/`inf` or climbs monotonically past progress≈0.15) → kill, classify research failure (do not retry).
- **No log output / no per-epoch `eval` line after ~120s** → kill, treat as infra failure (retry ≤2 per execute skill).
- **Wall-clock approaching 600s before completion** → the `timeout 600` enforces this; exit 124 = failure.
- **Severe throughput collapse** (projected epochs < ~110 from the early img/s read) → not an auto-abort (anneal still completes via the time-based schedule), but record as a likely under-annealing cause for analysis.
- **Dead-block guard** — the Milestone 1 gradient smoke (check (b)) MUST pass (`alpha.grad` nonzero) *before* the official run; if it fails, the block is not trainable and the run is meaningless — fix before launching.
- Note on smokes (resolves review #6): Milestone 1 runs a **fast in-process correctness/gradient smoke only** (compile + one forward/backward, seconds, NO training) — this is the pre-run gate. There is **no separate throughput *training* run**: the single official 300s run IS the experiment, and its throughput/early-convergence numbers are read live from its first epochs. The two are not in conflict — one is an in-process unit check, the other is the real run.

## Verification Protocol

### Verification Procedure
Baseline = **95.87%** (from `exp-index.sh baseline` on `04-results.tsv`; bar = 95.97%). After the run completes, evaluate the three goal necessary conditions in order (stop at first failure):

1. **Clean run within wall guard** — `grep -c "^best_test_acc:" run.log` == 1 and the process exited 0 (not 124). Extract `total_seconds` via `grep "^total_seconds:" run.log`; require < 600. Fail ⇒ crash.
   - Command: `grep "^best_test_acc:\|^training_seconds:\|^total_seconds:\|^num_epochs:\|^peak_vram_mb:\|^num_params:" run.log`. Empty `best_test_acc` ⇒ read `tail -n 50 run.log`.
2. **Full training budget + scope intact** — `training_seconds` ≥ 295 (grep above). Confirm `prepare.py` byte-unchanged: `git diff --quiet -- prepare.py && git diff --quiet autoresearch/maximize-cifar10-test-accuracy-dev -- prepare.py` (both exit 0). Confirm only `train.py` changed: `git diff --name-only autoresearch/maximize-cifar10-test-accuracy-dev` lists `train.py` only. **Diff-content scope (review #4 hardening):** `git diff autoresearch/maximize-cifar10-test-accuracy-dev -- train.py` and read it — assert the diff is limited to (i) the new `GatedResidual` class and (ii) the one-token `layer2` change; assert it does **not** touch `_forward_once`, `forward` (TTA logic), `compute_whitening_weight`, the eval call, the training loop, or any HP constant. `git diff --stat` should show only `train.py` with a small line count (~10 added). Timeout 30s.
3. **Improvement ≥ +0.1pp + genuineness** — parse `best_test_acc` (the `XX.XX` from `grep "^best_test_acc:" run.log`); **PASS iff ≥ 95.97**. Genuineness cross-check: `max` of all per-epoch `best:` values in the `eval ep` trace == the summary `best_test_acc` (came from `Eval.evaluate`, not fabricated) — `grep "eval ep" run.log | grep -oP 'best:\s*\K[0-9.]+' | sort -rn | head -1` equals the summary. **Reward-hack / leakage checks (review #4 hardening):** (a) exactly one eval call site — `grep -c "evaluator.evaluate(" train.py` == 1, called once per epoch (it is inside the per-epoch block, not the inner step loop); (b) seed integrity — `grep -c "manual_seed" train.py` == **2** (the existing `torch.manual_seed(42)` + `torch.cuda.manual_seed(42)`; no new seed line) and the local whitening `Generator().manual_seed(0)` is unchanged (`grep -c "Generator().manual_seed" train.py` == 1); (c) no test-set leakage in `train.py` — `grep -n "train=False\|test_set\|testset\|\.targets\b" train.py` returns nothing eval-related (only `datasets.CIFAR10(... train=True ...)` present); (d) `forward()` still does only flip-TTA (2 `_forward_once` passes) — confirmed by the diff-content check in step 2 (forward untouched), so no eval-circumvention was introduced.
   - Verdict: ≥95.97 with all above ⇒ **improvement**; valid run <95.97 ⇒ **no-improvement**; scope/seed/eval violation ⇒ **invalid**; crash/timeout ⇒ **crash**.

### Informational Metrics (Optional)
- `peak_vram_mb`: `grep "^peak_vram_mb:" run.log` — VRAM headroom (expect ~1.8–2.0 GB).
- `training_seconds` / `num_epochs` / `num_steps`: `grep "^training_seconds:\|^num_epochs:\|^num_steps:" run.log` — confirm full budget + epoch count (expect ~145–155, vs EXP-003's 174); fewer epochs is the capacity/throughput trade.
- `num_params`: `grep "^num_params:" run.log` — expect **7,784,627** total (EXP-003's 6,603,954 + 1,180,672 from the two new `conv_bn(256,256)` + 1 for the `α` gate; total includes the frozen 1,458-weight whiten conv). Learnable ≈ 7,783,169 (the smoke's check (e) value).
- Early trajectory: `grep "eval ep   1\|eval ep  10\|eval ep  25" run.log` — for the mechanism trace (identity init ⇒ ep1/ep10 should match EXP-003 within noise).
