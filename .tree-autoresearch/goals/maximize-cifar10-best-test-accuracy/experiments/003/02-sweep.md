# Sweep EXP-003: Balance CutMix and Drop Path
**Created**: 2026-08-05

## Operating Context

- **Machine / resources**: NVIDIA H20, physical GPU 0, 97,871 MiB total memory
- **Baseline operating point**: EXP-002 at `CUTMIX_PROB=0.50`, `MAX_DROP_PATH=0.08`, `best_test_acc=95.23%`
- **Other coupled config**: seed 42, 300-second charged training budget, batch size 256, CutMix alpha 1.0 and end fraction 0.75, identical PreAct WRN and evaluation harness

## Search Space & Optimizer & Budget (as used)

- **Optimizer**: grid via `itertools.product`
- **Direction**: maximize
- **Trials**: 6 grid points, 5 newly executed and 1 reused parent point (`MAX_PARALLEL=1`)
- **Parameters**:
  - `CUTMIX_PROB`: `{0.35, 0.50, 0.65}`
  - `MAX_DROP_PATH`: `{0.04, 0.08}`

## Trial Table

Full data: `experiments/003/trials.tsv`

| rank | trial_idx | `CUTMIX_PROB` | `MAX_DROP_PATH` | search `best_test_acc` |
|------|-----------|----------------|-----------------|------------------------|
| 1 | 5 | 0.65 | 0.08 | 95.48% |
| 2 | 4 | 0.65 | 0.04 | 95.42% |
| 3 | 2 | 0.50 | 0.04 | 95.28% |
| 4 | 3 | 0.50 | 0.08 | 95.23% (reused parent) |
| 5 | 0 | 0.35 | 0.04 | 95.20% |
| 6 | 1 | 0.35 | 0.08 | 94.86% |

## Confirmed Winner

- **Winning parameters**: none; no searched configuration passed confirmation
- **Primary metric**: best confirmed `best_test_acc=95.28%`, below the required 95.33% threshold and therefore not accepted
- **Necessary Conditions**: failed; trial 4 completed normally at 300.0 charged training seconds but improved only 0.05 points over the 95.23% parent
- **Informational Metrics**: trial 4 `final_test_acc=95.25%`, `final_test_loss=0.1963`, `training_seconds=300.0`, `total_seconds=468.3`, `startup_seconds=1.1`, `peak_vram_mb=1178.9`, `num_epochs=137`, `num_steps=26596`, `num_params=2748890`
- **Applied to**: no sweep candidate; `train.py` retains the EXP-002 parent values `CUTMIX_PROB=0.50` and `MAX_DROP_PATH=0.08`

## Errors & Dead Ends

- Trial 5 (`CUTMIX_PROB=0.65`, `MAX_DROP_PATH=0.08`) led the search at 95.48% but failed its single full confirmation: `best_test_acc=95.19%`, `final_test_acc=95.15%`, `final_test_loss=0.2029`, `training_seconds=300.0`, `total_seconds=464.5`, `startup_seconds=1.1`, `peak_vram_mb=1178.9`, `num_epochs=145`, `num_steps=28164`, and `num_params=2748890`. The confirmation is 0.04 points below the 95.23% parent and 0.14 points below the required 95.33% threshold. The 0.29-point search-to-confirmation regression is treated as winner's-curse evidence; the configuration will not be retried.
- Trial 4 (`CUTMIX_PROB=0.65`, `MAX_DROP_PATH=0.04`) ranked second in search at 95.42% but also failed its single full confirmation: `best_test_acc=95.28%`, `final_test_acc=95.25%`, `final_test_loss=0.1963`, `training_seconds=300.0`, `total_seconds=468.3`, `startup_seconds=1.1`, `peak_vram_mb=1178.9`, `num_epochs=137`, `num_steps=26596`, and `num_params=2748890`. Its 0.14-point search-to-confirmation regression left it 0.05 points below the required threshold, so it was rejected and not retried.
- Trial 2 was the next-ranked unconfirmed point at only 95.28% in search, already below the 95.33% necessary-condition threshold. The required Claude adversarial review rejected spending a confirmation on a configuration that could pass only by exceeding its selected search estimate through run-to-run variance. Lower-ranked points were likewise ineligible, so the candidate list was exhausted.

## Human Notes

> None.
