# Plan EXP-015: Mild policy-based augmentation (RandAugment / TrivialAugment), replacement design
- **Created**: 2026-06-30

Chosen idea: `01-brainstorm.md` § Chosen Idea + `proposals/idea-01.md`. Reviewed: `01-idea-review.md`.
Baseline (from `04-results.tsv`): **96.38** (EXP-008). Improvement bar: best cell beats the SAME-SESSION c0 control by **>0.1pp** (the ~0.1pp noise floor makes the stored 96.38 too weak to compare against directly).

## Hypothesis (testable)
A MILD RandAugment(1,6) — tested both REPLACING RandomErasing (cA) and ADDED on top of the full occlusion stack (cB), keeping Cutout12 — adds geometric+photometric diversity orthogonal to occlusion and raises `best_test_acc` to **≥ 96.48** (baseline 96.38 + the 0.1pp bar) AND clearly above the same-session control, without under-fitting (ep25 ≈ control; best ≠ final-still-climbing; num_epochs ~142–155; wall < 600s).
NEGATIVE-INFERENCE GUARD (review #7): cA/cB both keep Cutout12, so a tie/loss bounds only the policy-aug INCREMENT over the current occlusion recipe — it does NOT prove transform-augmentation is dead in general (untested: policy replacing Cutout, stronger/curriculum magnitude, TrivialAugment). The replace-vs-add pair (cA vs cB) disambiguates whether any policy-aug loss is over-regularization (cB add < cA replace) vs the mechanism being unhelpful here (both ≈ c0).

## Milestones

### Milestone 1: Code changes implemented + construction smoke-correct
- [ ] Add `import os` and `from torchvision.transforms import RandAugment` to train.py.
- [ ] Add `AUG_MODE = os.environ.get("AUG_MODE", "baseline")` plus `RANDAUG_N`/`RANDAUG_M` env reads (defaults N=1, M=6). `AUG_MODE ∈ {baseline, randaug_replace, randaug_add}`.
- [ ] Add a MODULE-LEVEL helper `build_train_tf(aug_mode, n, m)` returning the `transforms.Compose` (so it is unit-testable WITHOUT running `main()` — review #6). `main()` calls it.
- [ ] `build_train_tf` logic: `pre=[RandomCrop(32,4), RandomHorizontalFlip()]`; if mode startswith `randaug` insert `RandAugment(n,m)` (PIL) into `pre`; `post=[ToTensor(), Normalize(...), Cutout(12)]`; append `RandomErasing(...)` to post UNLESS mode==`randaug_replace`. (baseline & randaug_add keep RandomErasing; randaug_replace drops it.)
- [ ] Print `aug_mode` + `randaug_n`/`randaug_m` in the final summary block.
- [ ] `AUG_MODE=baseline` must reproduce the EXACT current pipeline (RandomCrop, Flip, ToTensor, Normalize, Cutout12, RandomErasing) — regression guard.
- [ ] Verify construction (review #6): `CUDA_VISIBLE_DEVICES=1 uv run python -c "import train; [train.build_train_tf(m,1,6) for m in ('baseline','randaug_replace','randaug_add')]; print('train_tf OK')"`; then `git diff --name-only` shows ONLY train.py.

### Milestone 2: Pre-smoke DataLoader-throughput probe (review #4; NO new file — review #5)
- [ ] INLINE probe via `uv run python -c "..."` (creates NO file — only train.py may exist as a code change): import `train.build_train_tf('randaug_add',1,6)`, build `datasets.CIFAR10` with it + the SAME DataLoader config (batch 512, num_workers 8, persistent_workers, prefetch_factor 4), warm up ~20 batches then time ~150 batches, print images/sec. Times the CPU aug pipeline WITHOUT touching the frozen 300s budget and WITHOUT adding a tracked/untracked helper script.
- [ ] Pass criterion: probe img/s ≳ 27k (must exceed the ~26k steady-state GPU compute rate so the loader is not the bottleneck). If < ~24k → CPU-bound risk; the run will inflate WALL `total_seconds` (not cut epochs) → watch `total_seconds` < 600 closely and note the fairness caveat (review #4: a policy cell using much more wall than c0 at equal compute budget is allowed by the harness but flagged).
- [ ] Confirm `nvidia-smi` shows GPU 1 idle (no foreign job) before the official run (infra-errors EXP-010/014).

### Milestone 3: Run the 3-cell same-session set
- [ ] c0 `AUG_MODE=baseline` → `run_c0.log` (same-session noise control)
- [ ] cA `AUG_MODE=randaug_replace RANDAUG_N=1 RANDAUG_M=6` → `run_cA.log` (RandAugment REPLACES RandomErasing)
- [ ] cB `AUG_MODE=randaug_add RANDAUG_N=1 RANDAUG_M=6` → `run_cB.log` (RandAugment ADDED on top of Cutout12+RandomErasing)
- [ ] Each: `CUDA_VISIBLE_DEVICES=1 AUG_MODE=... timeout 600 uv run train.py > run_<cell>.log 2>&1`, `nvidia-smi` logged immediately before into `gpu_<cell>.log`.
- [ ] All three must report num_epochs in ~142–155 and total_seconds < 600 (else contention/CPU-bottleneck → re-run the FULL set once GPU 1 idle; same-session controls only hold if all cells equally (un)contended).

### Milestone 4: Verdict
- [ ] Extract best_test_acc, final_test_acc, num_epochs, total_seconds, ep25 for all cells.
- [ ] PRIMARY (goal condition, review #2): best policy cell `best_test_acc` ≥ **96.48** (stored baseline 96.38 + 0.1pp).
- [ ] CONTEXT: also report margin over same-session c0 (noise control). A pass that clears c0 but not 96.48 is NOT a goal improvement.
- [ ] MULTIPLE-COMPARISON CAVEAT (review #3): with 2 policy cells at a ~0.1pp noise floor, a single cell clearing 96.48 by a hair via max(cA,cB) is treated as UNPROVEN unless the margin is clearly >noise OR corroborated by the other policy cell also lifting over c0. Note this in the analysis.
- [ ] UNDER-FIT DIAGNOSIS (review #8): if a policy cell loses, check ep25 vs c0 AND best-vs-final / late-epoch trend — best==final & still-climbing ⇒ under-fit (magnitude too high for 150ep), distinct from a true ceiling.
- [ ] ON A WIN (review #1): set the winning AUG_MODE/N/M as the DEFAULT in train.py (not env-gated) so the goal's frozen procedure `CUDA_VISIBLE_DEVICES=1 uv run train.py` reproduces the winning recipe before the result is committed.

## Code Changes
- **train.py** (the ONLY editable file):
  - Imports: add `import os`; add `from torchvision.transforms import RandAugment` (torchvision already imported — no new dependency; verified importable in torchvision 0.24.1).
  - Config block: `AUG_MODE = os.environ.get("AUG_MODE", "baseline")`; `RANDAUG_N = int(os.environ.get("RANDAUG_N", "1"))`; `RANDAUG_M = int(os.environ.get("RANDAUG_M", "6"))`. Valid AUG_MODE: `baseline` | `randaug_replace` | `randaug_add` (unknown value → raise/explicit fallback to baseline-equivalent, not silent).
  - NEW module-level helper `build_train_tf(aug_mode, n, m)` (unit-testable without `main()` — review #6):
    ```
    pre = [RandomCrop(32, padding=4), RandomHorizontalFlip()]
    if aug_mode in ("randaug_replace", "randaug_add"):
        pre.append(RandAugment(num_ops=n, magnitude=m))      # PIL in/out, before ToTensor
    post = [ToTensor(), Normalize(EVAL_MEAN, EVAL_STD), Cutout(12)]
    if aug_mode != "randaug_replace":                         # baseline & randaug_add keep RE
        post.append(RandomErasing(p=0.25, scale=(0.02,0.15), ratio=(0.3,3.3), value=0.0))
    return transforms.Compose(pre + post)
    ```
    `main()` replaces the static `train_tf = transforms.Compose([...])` (train.py:205) with `train_tf = build_train_tf(AUG_MODE, RANDAUG_N, RANDAUG_M)`.
  - Summary block: add `print(f"aug_mode: {AUG_MODE}")` and `print(f"randaug_n: {RANDAUG_N} | randaug_m: {RANDAUG_M}")`.
  - Why this tests the hypothesis: `randaug_replace` isolates policy-aug as a single-variable swap (RandomErasing→RandAugment); `randaug_add` isolates it as an addition; both vs same-session c0. Everything else (Cutout12, crop, flip, whitening, EMA, flip-TTA, optimizer, LR, seed) byte-identical.
  - Edge cases: (a) RandAugment expects PIL/uint8 — placement before ToTensor satisfies this (CIFAR10 yields PIL). (b) geometric ops fill exposed corners with 0 (default `fill=None`) in raw-pixel space → standard AutoAugment-CIFAR behavior, acceptable. (c) `AUG_MODE=baseline` MUST reproduce the exact current pipeline (Cutout12 + RandomErasing) — regression guard, M1. (d) ON A WIN the winning mode becomes the train.py DEFAULT (review #1) so bare `uv run train.py` reproduces it.

## Configuration Changes
- AUG_MODE: (new) baseline | randaug_replace | randaug_add — selects the augmentation configuration.
- RandomErasing: present (baseline, randaug_add) → REMOVED in randaug_replace (the replacement arm — avoids 3 stacked occlusion-like augs; idea-review #2). randaug_add KEEPS it to test policy-aug as a pure addition (plan-review #7 disambiguation).
- RandAugment magnitude: N=1, M=6 — deliberately ≪ the RandAugment CIFAR default (N=2, M=14) because our ~150ep budget is far shorter than the 200–2000ep canonical recipes (knowledge/references/policy-augmentation.md).
- No change to: model, optimizer, LR schedule, EMA, TTA, batch, Cutout12, whitening, seed (42).

## Execution Environment
- Method: local; each cell a SEPARATE `train.py` process. `CUDA_VISIBLE_DEVICES=1 AUG_MODE=<mode> [RANDAUG_N/M=...] timeout 600 uv run train.py > run_<cell>.log 2>&1`.
- Resources: single GPU (H20) on **GPU 1** (`CUDA_VISIBLE_DEVICES=1` — GPU 0 busy, per goal hard constraint + memory). VRAM trivial (~1.6GB).
- Estimated runtime: ~450s wall/cell × 3 ≈ 23 min + pre-smoke; each cell well under the 600s wall cap.
- Log output: `experiments/015/run_c0.log`, `run_cA.log`, `run_cB.log`; `nvidia-smi` snippet logged before each into the same logs or a `gpu_<cell>.log`.
- Tool skill: none (local run).

## Abort Criteria
- Any cell: img/s collapses (<18k sustained) → CPU-aug dataloader bottleneck; abort that variant, reduce to N=1 single-op or drop the cell.
- `nvidia-smi` shows a foreign GPU-1 job during a cell → contention (infra-errors EXP-010/014): mark logs `_contended`, re-run the FULL same-session set once GPU 1 idle.
- Any cell crashes (empty `best_test_acc:` grep) → read `tail -50 run_<cell>.log`; fix or drop.
- A policy cell shows divergence (test_acc stuck ~10–20% through mid-training) → magnitude far too high; record and abort that cell.
- Wall `total_seconds` ≥ 600 (timeout kill, exit 124) → infra failure for that cell.

## Verification Protocol

### Verification Procedure
1. Baseline: `bash .../exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv` → 96.38. Goal bar = **96.48** (baseline + 0.1pp). Same-session c0 is the NOISE control, not a substitute for the absolute bar (review #2).
2. Run all 3 cells per Milestone 3.
3. Extract per cell: `grep "^best_test_acc:\|^final_test_acc:\|^training_seconds:\|^total_seconds:\|^num_epochs:\|^aug_mode:" run_<cell>.log`. Empty `best_test_acc` ⇒ crash → `tail -50`.
4. ep25 + late-trend (under-fit check, review #8): `grep "eval ep  25" run_<cell>.log` for ep25; inspect the last ~5 `eval ep` lines for best-vs-final trend. Policy ep25 within ~0.5pp of c0 ep25 AND best ≠ final-still-climbing = healthy fit.
5. **Necessary conditions (goal file)**:
   - (a) Completes without crash, within budget, prints valid `best_test_acc`, wall < 600s (no exit 124). FAIL → no-improvement/crash.
   - (b) Best policy cell `best_test_acc` ≥ **96.48** (stored baseline + 0.1pp). Report margin over same-session c0 as noise context; apply the multiple-comparison caveat (review #3 — a hairline max(cA,cB) pass uncorroborated by the other cell is unproven). FAIL → no-improvement.
   - (c) Integrity: only train.py changed (`git diff --name-only`), prepare.py byte-unchanged (`git diff --quiet -- prepare.py`), ≤1 eval/epoch (eval cadence unchanged), seed fixed at 42 (no seed hacking). FAIL → invalid.
6. Same-session validity: all cells num_epochs ~142–155, no contention, total_seconds < 600; else re-run the full set (infra-errors EXP-010/014).
7. ON A WIN: bake the winning AUG_MODE/N/M as the train.py DEFAULT and re-confirm `CUDA_VISIBLE_DEVICES=1 uv run train.py` (bare, no env) reproduces the winning best_test_acc within noise before commit (review #1).
8. Cleanup: keep run logs in experiments/015/; no `run.log` left in the repo root.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run_<cell>.log`.
- num_epochs / num_steps / training_seconds / total_seconds: `grep` same — confirms full-budget use, throughput-free claim (epochs in band), and wall headroom to the 600s cap.
- num_params: `grep "^num_params:"` — unchanged across cells (aug-only change).
- ep25 test_acc per cell: under-fit diagnostic (the key signal distinguishing "saturated" from "under-fit" if policy cells lose).
