# Dissecting Supervised Contrastive Learning
- **Authors**: Florian Graf, Christoph Hofer, Marc Niethammer, Roland Kwitt
- **Venue**: ICML 2021
- **URL**: https://proceedings.mlr.press/v139/graf21a.html

## Key Contributions
- Shows that cross-entropy and supervised contrastive objectives share a regular-simplex class-collapse optimum under mild assumptions.
- Finds materially different optimization behavior even when the asymptotic representation geometry agrees.
- Links proximity to the simplex configuration with generalization in empirical studies.

## Relevance
The paper weakens the claim that an auxiliary SupCon term necessarily supplies a new endpoint geometry beyond CE. Its possible value here is optimization bias toward class-compact features, but EXP004 already uses strong CutMix and SAM; a single-view joint auxiliary loss needs direct evidence and careful dose control.

## Key Techniques
- Analyze within-class collapse and equiangular class-center geometry.
- Compare fitting dynamics of CE and supervised contrastive learning under label corruption.
