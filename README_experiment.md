# Stage 4 experiment: effect of training-data size on a sparse autoencoder

**Branch:** `experiment-datasize`
**Base:** the group's reproduction of Cunningham et al. (2023), *Sparse Autoencoders Find Highly Interpretable Features in Language Models* (arXiv:2309.08600). The reproduction itself is on `main`.

## Question

How much activation data does the SAE need before its held-out reconstruction error, its number of live features, and the interpretability of its features stop improving?

## Setup

**Fixed for every run** (same as the reproduction unless noted):

| Setting | Value |
|---|---|
| Model | Pythia-70M (frozen) |
| Layer | 3 (residual stream, `hidden_states[3]`) |
| Dictionary size | 2,048 features (R = 4), tied weights |
| Sparsity coefficient alpha | 1.0 |
| Optimiser | Adam, learning rate 1e-3, batch size 64 |
| Training steps | 60,000 per run (same for every data size) |
| Initialisation | same seed (0) for every run |
| Dataset | WikiText-103 (substitute for the Pile) |

**Varied:** the number of training token activations: 500, 2,000, 8,000, 32,000, 128,000 and 401,882. All subsets come from one fixed random ordering of the training tokens, so smaller subsets are contained in larger ones.

**Data split (new in this experiment):** 5,000 documents give 401,882 training activations (the same pool as the reproduction). A further 500 documents the SAE never trains on form the held-out set used for all evaluation.

**Metrics (on held-out documents):**
- MSE per token, and the fraction of variance unexplained
- L0: average number of active features per token (active means activation > 0.001)
- Dead features: features that never fire on any held-out token
- Train MSE on up to 20,000 tokens of each run's own training subset, to measure the train vs held-out gap

**Interpretability check:** for 3 sizes (500, 8,000, 401,882), 10 random live features each. For every feature the five contexts where it fires most strongly were listed with the activating token in `[[brackets]]`. The 30 items were shuffled and anonymised and scored 0 to 2 by one team member who did not know the data size. 2 = the examples clearly share one idea, 1 = partly, 0 = unclear or mixed.

## Files

| File | What it is |
|---|---|
| `sae_datasize_experiment.ipynb` | The whole experiment (run in Colab) |
| `datasize_results.csv` | One row per data size: held-out MSE, fraction unexplained, L0, dead features, train MSE |
| `datasize_main_plots.png` | Reconstruction error, dead features and L0 against data size |
| `datasize_convergence.png` | Held-out MSE during training, one curve per size |
| `datasize_interpretability.png` | Mean interpretability score per size |
| `scoring_sheet.csv` | The anonymised sheet that was scored |
| `scoring_sheet_scored.csv` | The same sheet with the scores filled in |
| `scoring_key.csv` | Which data size each item came from (opened only after scoring) |
| `sae_datasize_<size>.pt` | Trained SAE weights for each size (optional) |
| `PROVENANCE.md` | Who wrote what (see below) |

## How to run

1. Open `sae_datasize_experiment.ipynb` in Google Colab and choose **Runtime, then Change runtime type, then T4 GPU**.
2. Run the cells in order. Section 5 is a quick test (about a minute). Section 6 is the experiment itself.
3. Expect about 20 minutes in total: collecting activations takes a few minutes and each of the six training runs takes about 100 seconds.
4. Package versions are pinned in the first cell (`transformers==4.36.2`, `datasets==2.16.1`, `accelerate==0.25.0`). Pip may print dependency-conflict warnings in Colab. They did not affect the results.

To rerun the interpretability scoring, run section 8 to create a new `scoring_sheet.csv` and `scoring_key.csv`, fill in the scores, then run the aggregation cell. Do not look at the key until the scoring is finished.

## Results

| Training tokens | Held-out MSE | Fraction unexplained | L0 | Dead features (of 2,048) | Train MSE |
|---|---|---|---|---|---|
| 500 | 61.21 | 0.123 | 30.8 | 0 | 5.46 |
| 2,000 | 52.82 | 0.106 | 19.0 | 0 | 9.88 |
| 8,000 | 39.71 | 0.080 | 25.8 | 3 | 20.06 |
| 32,000 | 31.62 | 0.063 | 31.1 | 3 | 25.18 |
| 128,000 | 29.18 | 0.058 | 32.0 | 6 | 26.91 |
| 401,882 | 28.40 | 0.057 | 32.6 | 16 | 26.98 |

Interpretability (10 features per size, one scorer): mean score 1.7 at 500 tokens, 1.8 at 8,000 and 1.8 at 401,882; 80% of features scored as clear at every size.

**Summary of findings:**
- Held-out error falls steeply up to about 32,000 tokens and then flattens (from 128,000 to 401,882 tokens it improves by about 3%).
- Small datasets are memorised: train error is far below held-out error at 500 tokens, and nearly equal at full size.
- No detectable change in interpretability across the three sizes tested. The L0 and dead-feature differences are small and come from single runs, so we do not treat them as trends.

## Limitations

- One run (one seed) per data size.
- A fixed 60,000-step budget, which exaggerates the small-data gap because those models overfit as training continues.
- Interpretability scored on 10 features per size by one person.
- Dead features are counted on held-out text only, so the number can overstate truly unused features. It is not directly comparable to the dead-feature figure in the reproduction README, which used a different sample, threshold and training length.
- One model, one layer, one dataset and one value of alpha.

## Provenance

See `PROVENANCE.md`. In short:
- `SparseAutoencoder` and `sae_loss`: reused as-is from the group's reproduction notebook.
- `collect_activations`: adapted from the group's notebook (adds document ids so held-out documents can be separated).
- Training loop (fixed steps), evaluation, subset construction, plotting, scoring-sheet and aggregation code: written for this experiment.
- AI assistance: we took help from an AI assistant (Claude) for this experiment's code and for drafting this README. We ran the experiment, scored the features and checked the results ourselves.
