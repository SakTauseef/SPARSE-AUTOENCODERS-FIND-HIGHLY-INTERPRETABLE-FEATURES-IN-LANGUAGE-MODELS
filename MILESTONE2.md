# Milestone 2 — Experiments (branch `experiment/l1-sweep`)

**This branch holds all Milestone 2 experiments.** The `main` branch is our Milestone 1 submission and is left unchanged.

## What is here

| File | Purpose |
|---|---|
| `milestone2_experiment.ipynb` | The full Milestone 2 pipeline: activation caching, L1 sweep, trade-off plot, neuron/PCA/SAE comparison, blinded rating, deployment files |
| Result files (`sweep_results.csv`, `tradeoff.png`, …) | Added to this branch after the notebook is run |

## Setup (same as Milestone 1)

- **Model:** Pythia-70M (`EleutherAI/pythia-70m`), frozen
- **Activations:** residual stream after layer 3
- **Data:** WikiText-103 paragraphs longer than 20 characters, in dataset order. We use 20,000 paragraphs; the first 5,000 are exactly Milestone 1's. The last 10% are held out for evaluation only.
- **SAE:** tied weights, `c = ReLU(Mx + b)`, `x̂ = Mᵀc`, loss `||x − x̂||² + α||c||₁`, dictionary size 2048 (R = 4)

**Changes from Milestone 1 (and why):**
- **Position 0 of each sequence is dropped.** The first token has an unusually large activation that would distort training.
- **Activations are mean-centred and scaled** so the average squared length is 1. This makes reconstruction error read as "fraction of variance unexplained." Because of this, α values here are not on the same scale as Milestone 1's α = 1.0.
- **Loss is averaged per batch**, not summed over an epoch, so numbers are comparable across data sizes.

## Experiments

1. **Additional experiment (from our proposal): L1 coefficient sweep.** Seven log-spaced α values (0.001 to 1.0), with the same data, dictionary size, seed and epochs. Only α changes. For each run we report:
   - FVU (reconstruction error)
   - L0 (average active features per token)
   - dead features
   - LM loss recovered when the SAE's reconstruction is spliced back into the model

   The output is a sparsity-vs-reconstruction trade-off plot.
2. **Reproduction check: SAE vs. neurons vs. PCA.** The paper's main claim is that SAE features are more interpretable than the model's own basis directions and PCA directions. We take 15 random directions from each method, show their top-activating contexts, shuffle them with the method hidden, and each group member rates them independently. The result is "% interpretable" per method.

The 2–3 α values used for spot-checks are picked by a rule written in the notebook **before** looking at any features: L0 between 5 and 100 and loss recovered ≥ 70%.

## How to run

1. Open `milestone2_experiment.ipynb` in Google Colab.
2. Runtime → Change runtime type → T4 GPU.
3. Runtime → Run all (about 20–30 minutes). If a `WARNING` about the α range appears after section 6, widen `ALPHAS` and rerun section 6.
4. Download `sae_outputs.zip` at the end.
5. **Rating:** each member copies `rating_sheet_BLANK.csv` to `ratings_<name>.csv` and fills `interpretable` (y/n) and `label`, without discussing with the others. Upload all copies into `sae_outputs/` and run section 11.

## Outputs

| File | Contents |
|---|---|
| `sweep_results.csv` | One row per α: FVU, L0, dead fraction, LM loss, loss recovered |
| `sweep_training_history.csv` | Per-epoch training curves for every α |
| `tradeoff.png` | The main experiment figure |
| `rating_sheet_BLANK.csv`, `rating_key_DO_NOT_OPEN.csv` | Blinded rating sheet and its answer key |
| `rating_results.csv`, `rating_results.png` | % interpretable per method (after rating) |
| `sae_final.pt` | Final SAE weights + model/layer/normalization settings (used by the app) |
| `feature_examples.json` | Top 10 contexts per live feature (used by the app) |
| `sae_alpha*.pt` | Weights for every α in the sweep |

## Provenance

All code in this notebook was written by our group, implementing the paper's equations. The SAE class and training loop extend Milestone 1's code. The evaluation metrics, baseline comparison and rating procedure are new for Milestone 2. No code was copied from the official repository (`HoagyC/sparse_coding`).
