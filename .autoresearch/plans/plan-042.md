# Plan EXP-042: torch.compile reduce-overhead + T_max=43
## Code Changes
1. `COSINE_T_MAX = 49` → `COSINE_T_MAX = 43`
2. `torch.compile(model)` → `torch.compile(model, mode="reduce-overhead")`
