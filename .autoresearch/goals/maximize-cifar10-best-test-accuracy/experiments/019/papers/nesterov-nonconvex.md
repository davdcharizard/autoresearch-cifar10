# Accelerated Gradient Descent Escapes Saddle Points Faster than Gradient Descent
- **Authors**: Chi Jin, Praneeth Netrapalli, Michael I. Jordan
- **Venue**: COLT 2018
- **URL**: https://proceedings.mlr.press/v75/jin18a.html

## Key Contributions
- Gives a Nesterov-style accelerated method a faster second-order-stationarity rate than gradient descent in general nonconvex optimization.
- Establishes direct single-loop acceleration without nested or proximal machinery.
- Uses a Hamiltonian argument and an improve-or-localize framework to track long-horizon behavior.

## Relevance

The theorem concerns a specific accelerated method rather than PyTorch SGD's `nesterov=True` recurrence, and stationarity is not CIFAR generalization. It nevertheless supports the narrow claim that a current-gradient correction can alter nonconvex exploration rather than merely add implementation overhead. Local production-batch trajectory gates remain essential because PyTorch's first update is 1.9 times ordinary momentum at momentum 0.9.

## Key Techniques
- Combine momentum history with a current-gradient correction.
- Analyze escape from saddles and long-run localization.
- Keep acceleration single-loop.

