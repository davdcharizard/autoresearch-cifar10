# Plan EXP-019: Squeeze-Excitation channel attention (layer2+layer3 residual branches)
- **Created**: 2026-06-30

## Summary
Add lightweight Squeeze-Excitation (SE) channel-attention blocks to the residual branches of the whitened ResNet-9, the first **channel-attention** lever on this goal — a genuinely new functional form (content-adaptive per-channel recalibration) orthogonal to every saturated axis (width/depth EXP-007/014, optimizer 009/010, input-aug 008/011/015, wd/LS 012, SAM 013, BN-noise 016/017, anti-aliasing 018). Identity-init the gate (`2*sigmoid` + zero-init `fc2` → gate=1.0 at init) so the validated recipe is bit-unperturbed at init even in the un-gated `Residual` blocks. Throughput-neutral by design (GAP + two 1×1 convs on a [N,C,1,1] tensor). Tested same-session against a no-SE control, with the EXP-016/017/018 hardened protocol (num_epochs≥135 gate, anti-gaming integrity, confirmation re-run on any apparent win).

Baseline (current): **96.38** (commit 07c3760). Improvement bar: **best_test_acc ≥ 96.48** (baseline +0.1pp) AND clearly > same-session c0 by >0.1pp, replicated on a confirmation re-run.

## Milestones

### Milestone 1: SE implemented + local smokes pass
- [ ] Add `import os` and env reads `SE_RATIO` (int, default 16), `SE_LAYERS` (str, default "" = baseline) near the hyperparameter block.
- [ ] Add `class SE(nn.Module)` — GAP → `Conv2d(c, cr, 1)` → ReLU → `Conv2d(cr, c, 1)` → `2*sigmoid` gate → channel rescale, where `cr = max(8, c // r)`. Zero-init `fc2` inside `__init__` too (defense-in-depth; the load-bearing re-zero is post-`apply`).
- [ ] Thread an optional `use_se` / `se_ratio` into `Residual` and `GatedResidual`; insert SE after `c2`, before the residual add (for `GatedResidual`, SE sits INSIDE the `alpha`-gate).
- [ ] Make `se_layers`/`se_ratio` **constructor args of `ResNet9`** (defaulting to the module globals `SE_LAYERS`/`SE_RATIO`): `def __init__(self, num_classes=10, scale_out=SCALE_OUT, se_layers=SE_LAYERS, se_ratio=SE_RATIO)`. `main()` keeps calling `ResNet9(NUM_CLASSES)` → picks up env defaults; smokes can construct variants in-process WITHOUT env/import juggling (fixes the import-time-global test problem).
- [ ] Wire `se_layers` digits {1,2,3} → layer1/layer2/layer3 in `ResNet9.__init__`.
- [ ] After `self.apply(self._weights_init)`, zero-init every `SE.fc2` weight AND bias (kaiming from `apply` clobbers the in-`__init__` zero; this re-assert is the load-bearing identity-init).
- [ ] Smoke A (MODEL-LEVEL identity-init — the real clobber guard): build `ResNet9(se_layers="123")`, feed a random [4,C,H,W] tensor to each SE submodule (C/H/W per stage), assert `torch.equal(se(x), x)` — i.e. gate==1.0 AFTER the post-`apply` zero-init has run. (A fresh `SE(c)` alone would be kaiming-clobbered by `apply`; the model-level check is what matters.)
- [ ] Smoke B (regression guard): `ResNet9(se_layers="")` has `num_params == 7,784,627` (baseline) and zero `SE` submodules.
- [ ] Smoke C (params/structure): `ResNet9(se_layers="23", se_ratio=16)` adds ~41k SE params (layer2 ~8.4k + layer3 ~33k) and exactly 2 SE submodules; `se_layers="123"` adds 3.
- [ ] Smoke D (finite fwd/bwd + SE trainable): one autocast bf16 forward+backward on a (512,3,32,32) channels_last batch is finite; `fc1.weight.grad` and `fc2.weight.grad` are non-None and finite for each SE block.
- [ ] Smoke script lives OUTSIDE the repo at `/tmp/exp019_smoke.py` (NOT in the working tree — keeps the train.py-only integrity check clean) and imports train via `PYTHONPATH=<root>`. Verify: `cd <root> && PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 uv run python /tmp/exp019_smoke.py` prints all PASS, then `git status --porcelain` shows only `train.py`.

### Milestone 2: Throughput probe (under-anneal gate pre-check)
- [ ] Probe script at `/tmp/exp019_probe.py` (outside repo). Time ~40 warm full train-steps (fwd+bwd+opt, autocast bf16, channels_last, sync each step) for `ResNet9(se_layers="")` (c0) and `ResNet9(se_layers="23")` (cA), and also `se_layers="123"` (cB) on GPU 1.
- [ ] Verify: each SE config's img/s ≥ ~0.92× c0 img/s (predicts num_epochs ≥ ~138). If any drops below the band that would push num_epochs < 135, note it; the official runs' printed `num_epochs` is the authoritative gate (probe is advisory).

### Milestone 3: Official same-session cells run on GPU 1
- [ ] Start a background `nvidia-smi` sampler (e.g. `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -l 5 > /tmp/exp019_smi.log &`) for the whole session, AND log a point `nvidia-smi` immediately before EACH cell (mid-run contention guard, infra-errors EXP-010/014).
- [ ] c0 (control): `SE_LAYERS=""` — baseline, full-speed same-session anchor.
- [ ] cA (**PRIMARY — the chosen hypothesis; this cell alone determines the verdict**): `SE_LAYERS="23" SE_RATIO=16` — SE at layer2+layer3.
- [ ] cB (**DIAGNOSTIC ONLY — informational, cannot by itself trigger an `improvement` verdict**): `SE_LAYERS="123" SE_RATIO=16` — SE at all three blocks, to learn whether early-layer channel attention helps. If cB beats c0 but cA does not, that is a NEW hypothesis → its own future confirmation experiment, NOT a credited EXP-019 win (avoids placement-search on the test metric).
- [ ] Each: `CUDA_VISIBLE_DEVICES=1 SE_LAYERS=... SE_RATIO=... uv run train.py > run_{cell}.log 2>&1`, wall < 600s.
- [ ] Record best_test_acc, num_epochs, ep25 (proxy via early eval lines), training_seconds, total_seconds, peak_vram_mb, num_params per cell; cross-check the per-step `img/s` trace in each log for a mid-run drop.

### Milestone 4: Verification + confirmation
- [ ] All cells fit num_epochs ≥ 135 AND all equally uncontended. Contention check is two-pronged: the background `nvidia-smi` sampler (no foreign PID / no util spike during any cell) AND the in-log `img/s` trace (no mid-run drop below the clean ~25k band). If EITHER signals contention on ANY cell, classify that cell as infra-confounded (`crash`/re-run) — NEVER collapse a contention-driven under-anneal into `no-improvement` — and re-run the FULL same-session set once GPU 1 is idle.
- [ ] Verdict keyed on **cA vs c0**: if cA ≥ 96.48 AND cA − c0 > 0.1pp, run a **confirmation re-run** of {cA, fresh c0} as a second same-session pair; require the delta to replicate (>0.1pp) before calling it an improvement.
- [ ] If the cA−c0 delta is <0.15pp on the first pass, treat as no-signal unless the confirmation pair replicates (low-c0-draw lesson, EXP-016/017).

## Code Changes
- **train.py** (the only editable file):
  - **Env reads** (near hyperparameters): `import os`; `SE_RATIO = int(os.environ.get("SE_RATIO", "16"))`; `SE_LAYERS = os.environ.get("SE_LAYERS", "")`. Empty `SE_LAYERS` ⇒ no SE ⇒ baseline behavior (regression-safe default).
  - **`class SE(nn.Module)`**:
    ```python
    class SE(nn.Module):
        """Squeeze-Excitation channel attention (Hu et al. 2018). Identity at init:
        fc2 is zero-init and the gate is 2*sigmoid(0)=1.0, so the branch is
        unperturbed at init even in the un-ReZero'd Residual blocks."""
        def __init__(self, c, r=16):
            super().__init__()
            cr = max(8, c // r)
            self.fc1 = nn.Conv2d(c, cr, 1)
            self.fc2 = nn.Conv2d(cr, c, 1)
            init.zeros_(self.fc2.weight); init.zeros_(self.fc2.bias)  # defense-in-depth
        def forward(self, x):
            s = x.mean((2, 3), keepdim=True)
            s = 2.0 * torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
            return x * s
    ```
  - **`Residual` / `GatedResidual`**: add `use_se=False, se_ratio=16` params; `self.se = SE(c, se_ratio) if use_se else None`; in forward, compute `out = self.c2(self.c1(x))`, then `if self.se is not None: out = self.se(out)`, then add (`x + out` for `Residual`, `x + self.alpha * out` for `GatedResidual`).
  - **`ResNet9.__init__(self, num_classes=10, scale_out=SCALE_OUT, se_layers=SE_LAYERS, se_ratio=SE_RATIO)`**: compute `seN = str(d) in str(se_layers)` for d in 1/2/3; pass `use_se=seN, se_ratio=se_ratio` into the `Residual(128)`/`GatedResidual(256)`/`Residual(512)` constructors. `se_layers`/`se_ratio` are constructor args (defaulting to the env globals) so the official runs read env while in-process smokes pass explicit values — no import-time-global juggling.
  - **Identity-init enforcement**: after `self.apply(self._weights_init)`, add
    ```python
    for m in self.modules():
        if isinstance(m, SE):
            init.zeros_(m.fc2.weight); init.zeros_(m.fc2.bias)
    ```
    (kaiming from `apply` hits the 1×1 convs; this re-asserts the zero-init on `fc2` so the gate starts at exactly 1.0).
- **Why this tests the hypothesis**: SE adds content-adaptive cross-channel recalibration — a modeling capability absent from all saturated axes — at <1% params and near-zero compute, so any accuracy lift is attributable to the new functional form, not to capacity/epochs (which EXP-014 proved are worth ≈0 here).
- **Risks / edge cases**:
  - *Identity-init clobber* (handled): `apply` runs after submodule construction → must re-zero `fc2` post-apply. Smoke A guards this.
  - *Throughput*: per-block GAP adds reduction kernels; `x.mean((2,3))` is async (no `.item()`/host sync), so the cost is small, but verified by the probe + the num_epochs≥135 gate.
  - *EMA*: SE has no buffers; `AveragedModel(use_buffers=True)` averages SE conv weights normally. Flip-TTA stays valid (GAP is flip-invariant over spatial dims).
  - *bf16/channels_last*: `mean`/`sigmoid`/1×1-conv all preserve memory format and run under autocast.

## Configuration Changes
- `SE_RATIO`: new env, default `16` (Hu et al. canonical bottleneck ratio). cB keeps 16 (probing placement, not capacity — SE params are already negligible, so r=8 would add little signal vs adding layer1).
- `SE_LAYERS`: new env, default `""`. Cells use `""` (c0), `"23"` (cA primary), `"123"` (cB).
- No change to PEAK_LR/WD/LS/EMA/aug/schedule — SE is identity at init so the validated recipe needs no retune (same rationale as the ReZero α=0 block, EXP-004).

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=1 SE_LAYERS=... SE_RATIO=... uv run train.py > run_{cell}.log 2>&1` from project root.
- Resources: single NVIDIA H20, **GPU 1** (mandatory — GPU 0 in use). VRAM trivial (~1.6 GB baseline; SE adds <1%).
- Estimated runtime: ~300s training + ~150s eval/startup per cell ≈ 7–9 min wall each; 3 cells + smokes + probe + (conditional) confirmation pair ≈ 45–70 min total.
- Log output: per-cell `run_{cell}.log` in project root, parsed via `grep`; deleted after recording.
- Tool skill: none (local run).

## Abort Criteria
- A foreign job appears on GPU 1 (`nvidia-smi` shows non-our PID at high util / multi-GB) → throughput drops below the ~25k img/s band → kill and re-run the FULL same-session set once GPU 1 is idle (epochs not comparable across unequal contention; infra-errors EXP-010/014).
- Any cell's loss goes NaN/inf or eval accuracy collapses (<50% past ep20) → kill that cell, inspect (SE init / gate bug).
- A run exceeds 600s wall-clock → killed, treated as failure (goal hard cap).
- No log output / no eval line after ~120s → silent hang; kill and inspect.
- num_epochs < 135 on an SE cell while c0 sits at ~150 → SE is under-annealing (throughput regression); record as under-anneal, do not credit any tie/loss as a clean ceiling result.

## Verification Protocol

### Verification Procedure
Baseline = **96.38** (from `exp-index.sh baseline`). Bar = **96.48** (baseline + 0.1pp). Run cells in order c0 → cA → cB in one session on GPU 1.

1. **Completion + budget (necessary)**: for each cell, `grep "^best_test_acc:\|^training_seconds:\|^total_seconds:\|^num_epochs:\|^peak_vram_mb:\|^num_params:" run_{cell}.log`. Empty `best_test_acc` ⇒ crash (read `tail -n 50`). Confirm `total_seconds` < 600. Pass = valid `best_test_acc` printed and wall < 600s. Timeout: kill any run exceeding 600s wall.
2. **Under-anneal gate (necessary for a trustworthy comparison)**: each cell `num_epochs ≥ 135` AND within the clean ~142–154 band for c0; all cells equally uncontended via BOTH the background `nvidia-smi` sampler (`/tmp/exp019_smi.log`: no foreign PID / util spike) AND the in-log per-step `img/s` trace (no mid-run drop below ~25k). If an SE cell is < 135 while c0 ≈ 150, OR either contention signal fires, the result is infra-confounded — classify `crash`/re-run the FULL set once GPU 1 is idle; do NOT collapse it into `no-improvement`.
3. **Primary metric (necessary) — keyed on cA only**: `cA.best_test_acc` ≥ 96.48 AND cA > same-session c0 by > 0.1pp. Compute `delta = cA − c0`. If `cA < 96.48` OR `delta ≤ 0.1` ⇒ no-improvement. **cB is diagnostic/informational only** — a cB-only win (cB beats c0 but cA does not) is recorded as a future-hypothesis lead, NOT an EXP-019 improvement (prevents placement-search on the test metric).
4. **Confirmation re-run (necessary on an apparent cA win)**: if condition 3 passes, re-run {cA, fresh c0} as a second same-session pair; require the >0.1pp delta to replicate. A win on pass 1 that does not replicate ⇒ no-improvement (low-c0-draw lesson, EXP-016/017).
5. **ep25 sanity (necessary for a clean read)**: SE cells' early-epoch accuracy (first eval line near 25 epochs) within ~0.5pp of c0 — confirms the identity-init did not depress early convergence. A depressed ep25 with a tie ⇒ init bug, not a clean null.
6. **Integrity / anti-gaming (necessary, ALWAYS run)**:
   - `git diff --quiet -- prepare.py` (exit 0 = frozen eval untouched). Also confirm only `train.py` changed (`git status --porcelain` lists only train.py).
   - Exactly one `evaluator.evaluate` call per epoch (unchanged from baseline; grep the loop — eval cadence not modified).
   - Seed 42 unchanged; no seed search.
   - **summary best == per-epoch max**: cross-check the printed `best_test_acc:` equals the max `test_acc` over the per-epoch `eval ep` lines (`grep "eval ep" run_{cell}.log | awk` max vs the summary line). A mismatch ⇒ invalid.
   - c0 has zero SE submodules (regression guard, Smoke B); SE only via the documented env path.

Render: **improvement** only if conditions 1–6 all pass for **cA** AND the cA confirmation pair replicates; else **no-improvement** (a cB-only signal is logged as a next-loop lead, not a win); **invalid** on any integrity failure (prepare.py touched, >1 eval/epoch, seed hack, summary≠per-epoch-max); **crash** on no metrics or unresolved GPU-1 contention.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run_{cell}.log` — confirms VRAM soft-constraint headroom (expect ~1.6 GB).
- num_epochs / num_steps / training_seconds: `grep` — confirms full-budget use and the under-anneal gate (condition 2).
- num_params: `grep "^num_params:" run_{cell}.log` — confirms SE param overhead (<1%); c0 must read 7,784,627.
- ep25 trajectory: first few `eval ep` lines — early-convergence health under the identity-init.
