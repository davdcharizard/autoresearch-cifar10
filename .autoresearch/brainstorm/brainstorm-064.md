# EXP-064: EMA decay 0.998 on winning BF16+CL config
With 60 epochs (vs 49 originally), EMA has more update steps. Faster tracking (0.998 vs 0.999) keeps EMA closer to the training model.
