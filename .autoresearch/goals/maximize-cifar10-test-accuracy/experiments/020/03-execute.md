# EXP-020: Cosine one-cycle decay (vs linear triangular LR shape)

## Execution

Overall Status & Info:
- **Created**: 2026-06-30
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-020
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (verdict no-improvement — see 04-analysis.md)

## Implementation Notes

### Summary
Implemented the cosine LR decay per the plan, all in `train.py`. Added `import math`/`import os`; added `SCHEDULE` env (default `tri` = baseline) and made `PCT_START` env-overridable (default 0.15). In the LR block (train.py:286-294), inserted a `cos` branch (`q=(progress−PCT_START)/(1−PCT_START); lr=PEAK·0.5·(1+cos(πq))`) while keeping the `tri` else-branch's expression byte-identical to the original linear formula — so `SCHEDULE=tri` is bit-for-bit baseline. Milestone 1 smokes (LR-trace + 2-step crash guard) all passed; smoke script in `/tmp` (only train.py shows in `git status`).

### Surprises & Discoveries
- LR-trace confirmed the intended shape difference quantitatively: cosine holds LR HIGHER than linear early-post-warmup (p=0.30: 0.370 vs 0.329), crosses near mid-decay, and runs LOWER late (p=0.90: 0.0135 vs 0.047), spending ~2.8× more of the post-warmup window below 0.05·PEAK (14.4% vs 5.1%). This is exactly the "more time in the low-LR tail" mechanism the hypothesis rests on.

### Decisions
- **cB = `SCHEDULE=cos PCT_START=0.10`** (shorter warmup, diagnostic-only) per plan — the verdict is keyed on cA (`cos`, PCT_START=0.15, single-variable vs c0). cB cannot trigger an improvement verdict (no schedule-search on the test metric; EXP-019 cB rule).
- Default `SCHEDULE` left at `tri` so the bare `uv run train.py` stays baseline; the on-win BAKE step (flip default to `cos`) happens only in analyze if cA wins.

## Experimental Adjustments

## Run Log

### Run 1 — Official same-session cells c0/cA/cB (Milestone 2)

Metadata:
- **Job ID**: local background, GPU 1
- **Log file(s)**: run_c0.log / run_cA.log / run_cB.log (project root, gitignored, deleted after recording)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-30
- **Ended**: 2026-06-30

Description:
- Three full 300s-budget runs in one session on GPU 1 under `timeout 600`: c0 (`SCHEDULE=tri`, control/regression), cA (`SCHEDULE=cos`, primary/verdict-bearing), cB (`SCHEDULE=cos PCT_START=0.10`, diagnostic). Background `nvidia-smi -l 5` sampler (/tmp/exp020_smi.log) for contention. The change is throughput-free, so all three should fit ~150 epochs near c0; testing whether cosine lifts cA ≥ 96.48 AND > c0 by >0.1pp.

Observations:
- All three cells ran clean to 150 epochs (throughput-free confirmed; the cos branch adds no per-step cost). img/s ~25.3–25.4k all cells, well above the pre-registered 22000 contention floor. GPU1 mem 3843→6016 MiB (all ours); 0 foreign GPU0-util samples. Integrity intact: best==per-epoch-max all cells; `git status` shows only ` M train.py`.

Key Metrics:
- **c0 (tri)**: best 96.04 @150ep, final 95.93, 455.1s, peak_vram 1635 MB, per-epoch-max 96.04 ✓
- **cA (cos)**: best 96.36 @150ep, final 96.36, 435.1s, per-epoch-max 96.36 ✓ → **+0.32pp vs c0**
- **cB (cos, PCT_START=0.10)**: best 96.13 @150ep, final 96.10, per-epoch-max 96.13 → +0.09pp vs c0 (diagnostic; shorter warmup underperforms cA's 0.15 → 0.15 is the better cosine operating point)
- cA's +0.32 same-session lead is strong BUT absolute 96.36 < 96.48 floor, and c0 drew anomalously low (96.04 vs stored baseline 96.38) — the recurring low-control-draw pattern (cf. EXP-019). Confirmation pair required before verdict.

### Run 2 — Confirmation pair {fresh c0b, cAb} (verdict-determining replication)

Metadata:
- **Job ID**: local background (detached `/tmp/exp020_confirm.sh`), GPU 1
- **Log file(s)**: run_c0b.log / run_cAb.log (project root, gitignored, deleted after recording); sampler /tmp/exp020_smi2.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-30 11:13
- **Ended**: 2026-06-30 11:28

Description:
- Second same-session pair {c0b (`SCHEDULE=tri`), cAb (`SCHEDULE=cos`)} on GPU 1 under `timeout 600`, mirroring the EXP-019 confirmation approach. Tests whether the Run-1 +0.32pp same-session cosine delta REPLICATES and whether a normal-draw control lets cosine clear 96.48. Per plan, verdict is keyed on cAb: improvement requires cAb ≥ 96.48 AND a replicated cAb−c0b > 0.1pp.

Observations:
- Both cells ran clean to 150 epochs; img/s ~25.3–25.5k (no throughput loss vs Run 1). GPU1 mem 3843→6016 MiB (all ours). One single GPU0-util>0 sample across the whole session = foreign background blip on the shared node; our GPU1 runs unaffected (full img/s + full epoch count both cells). Integrity intact: best==per-epoch-max both cells; `git status` shows only ` M train.py`.
- **The cosine lead COLLAPSED**: with a NORMAL control draw (c0b 96.35, close to stored baseline 96.38), the +0.32 same-session delta dropped to +0.04. This confirms Run-1's +0.32 was a low-control-draw artifact (c0 anomalously low at 96.04), not a real schedule effect — the exact EXP-019 SE pattern.

Key Metrics:
- **c0b (tri)**: best 96.35 @150ep, final 96.30, 444.2s, peak_vram 1635 MB, per-epoch-max 96.35 ✓
- **cAb (cos)**: best 96.39 @150ep, final 96.30, 431.5s, per-epoch-max 96.39 ✓ → **+0.04pp vs c0b**
- Cross-session: cos pair means {96.36, 96.39}; tri pair means {96.04, 96.35}. The cosine delta did NOT replicate (+0.32 → +0.04, < 0.1pp bar) and cAb 96.39 < 96.48 floor → both gating conditions fail.

## Verification Results

### Conditions Checked

- **NC1 — primary metric ≥ baseline+0.1 (≥ 96.48) AND clearly above same-session control beyond noise, replicated**: **FAILED**.
  - Verdict-bearing cell cAb (cos) = **96.39%** < 96.48 floor → fails absolute bar.
  - Same-session delta did NOT replicate: Run-1 cA−c0 = +0.32pp but Run-2 cAb−c0b = +0.04pp (< 0.1pp). The Run-1 lead was a low-control-draw artifact (c0 96.04 vs c0b 96.35).
  - Source: run_cA.log / run_c0.log (Run 1), run_cAb.log / run_c0b.log (Run 2).
- **Integrity / hard-constraints**: PASS — only `train.py` modified; `SCHEDULE=tri` is byte-identical baseline; best==per-epoch-max all 5 cells (no eval-cache/seed anomaly); num_epochs=150 all cells (no under-anneal, no throughput hack); single GPU0 blip immaterial (img/s ≥ 25.3k throughout).

### Informational Metrics

- num_epochs: 150 (all 5 cells) — change is throughput-free as predicted.
- peak_vram: 1635 MB (unchanged vs baseline).
- cB diagnostic: shorter cosine warmup (PCT_START=0.10) underperforms 0.15 → no shorter-warmup lead to carry forward.

## Errors & Dead Ends

### 2026-06-30 — confirm runner backgrounded `&` inside a `run_in_background` Bash wrapper
- Error: the wrapper Bash call returned "completed" immediately while `/tmp/exp020_confirm.sh` (launched with trailing `&`) kept running detached; an early metric grep saw a half-written c0b log (ep 35) and misread it as a killed run.
- Root cause: double-backgrounding — `&` inside an already-`run_in_background:true` call detaches the script from the wrapper's lifecycle, so the completion notification fires on the wrapper, not the work.
- Resolution: verified liveness via `pgrep`/log tail, then polled for the script's own `CONFIRM DONE` sentinel before reading metrics. No data lost; both cells completed cleanly.
- Do NOT: trust the task-completion notification when the launched command itself backgrounds with `&`. Either drop the inner `&` (let `run_in_background` own the lifecycle) or gate on an explicit completion sentinel in the script's output.

## Human Notes

>
