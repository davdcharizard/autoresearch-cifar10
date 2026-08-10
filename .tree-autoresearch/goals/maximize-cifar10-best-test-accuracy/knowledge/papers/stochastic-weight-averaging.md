# Averaging Weights Leads to Wider Optima and Better Generalization
- **Authors**: Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
- **Venue**: UAI 2018
- **URL**: https://www.auai.org/uai2018/proceedings/papers/313.pdf

## Key Contributions
- Averages later SGD trajectory points into one model and improves CIFAR residual-network generalization with little overhead.
- Connects useful averaging to a learning-rate regime that maintains trajectory diversity.

## Relevance
Weight averaging is a low-cost option when late iterates remain diverse. Under a strongly decaying cosine schedule, verify that the averaging window is not effectively identical to the final weights and handle BatchNorm state explicitly.

## Key Techniques
- Uniform or exponential parameter averaging.
- Constant/cyclic learning-rate tails.
- BatchNorm-statistics management for the averaged model.
