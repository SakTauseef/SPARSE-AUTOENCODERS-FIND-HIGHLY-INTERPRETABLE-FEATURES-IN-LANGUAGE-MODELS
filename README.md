# Sparse Autoencoder Reproduction

Reproduction of: Cunningham, Ewart, Riggs, Huben & Sharkey (2023), *"Sparse Autoencoders Find Highly Interpretable Features in Language Models"*, arXiv:2309.08600.

Official reference repo (used for reference only, not copied): https://github.com/HoagyC/sparse_coding

## What the paper does

Neurons inside language models are often *polysemantic* — a single neuron reacts to several unrelated concepts, which makes the model hard to interpret. The paper's hypothesis is that this happens because of *superposition*: the model represents more concepts than it has neurons, so it packs multiple concepts into overlapping directions, relying on those concepts rarely firing at the same time.

The proposed fix: freeze the trained language model, record its internal activations, and train a separate **sparse autoencoder** on those activations. The autoencoder learns a larger set of "dictionary features" that, empirically, tend to be far more interpretable — each one closer to representing a single clean concept — than the original neurons.

## What we reproduced (reduced scope)

Given the compute budget for this assignment, we reproduce the core mechanism at a much smaller scale than the paper:

| | Paper | This reproduction |
|---|---|---|
| Model | Pythia-70M / Pythia-410M | Pythia-70M |
| Dataset | The Pile | WikiText-103 (substitute — see note below) |
| Activation source | residual stream / MLP / attention | residual stream, layer 3 |
| Dictionary size ratio (R) | swept, e.g. 0.5–8x | 4x (dict_size = 2048) |
| Evaluation | GPT-4 autointerpretability + IOI causal patching | manual inspection of top-activating text per feature |

**Dataset substitution note:** the paper collects activations by running the model over the Pile. We substitute WikiText-103, a smaller and more accessible general-text corpus. This is a comparable-domain substitute (general English text), but is narrower in topic diversity than the Pile, which likely affects how many distinct, clean concepts the autoencoder can discover from a limited sample.

## Method

Implements the paper's equations directly:

```
c  = ReLU(Mx + b)             # encoder
x̂  = Mᵀc                      # decoder (tied weights)
L  = ||x − x̂||² + α||c||₁     # reconstruction + sparsity penalty
```

Pipeline: freeze Pythia-70M → run WikiText-103 samples through it → record layer-3 residual stream activations → train the autoencoder above on those activations → inspect learned features by finding which real text snippets activate each one most strongly.

## Design choices and why

- **Pythia-70M, layer 3**: smallest model in the family, keeps every run fast enough to iterate on a free-tier Colab GPU.
- **Dictionary ratio R = 4**: gives the autoencoder 4x more "drawers" than the model has switches, following the paper's approach of over-completing the dictionary.
- **Tied encoder/decoder weights, row-normalized**: matches the paper's stated architecture (Appendix B), and keeps the parameter count and training stable.
- **Manual interpretability check instead of GPT-4 autointerpretability**: the paper's automated scoring pipeline requires GPT-4 API calls at scale; for this reduced reproduction we instead manually inspect the top-5 activating text snippets per feature, which is a legitimate (if less rigorous) proxy for "does this feature mean one consistent thing."

## Current results

### Reconstruction is learning correctly

Training on a small sanity-check subset (500 activation vectors), reconstruction loss drops steadily and doesn't stall:

```
epoch   0 | recon 3364.0
epoch  10 | recon  161.8
epoch  20 | recon   84.3
epoch  29 | recon   78.6
```

This confirms the encoder/decoder implementation is correct.

### Sparsity is not yet strong enough

The sparsity term plateaus rather than continuing to fall, and the fraction of "open drawers" per token stays far above the target (a genuinely sparse result should be roughly 1–2% of features active per token, not the 20–26% we currently see):

| alpha | final recon | final sparsity | avg active features / token |
|---|---|---|---|
| 0.001 | 61.1 | 2070.8 | 534.4 (26.1%) |
| 0.01  | 68.8 | 2048.7 | 529.4 (25.9%) |
| 0.05  | 77.0 | 1903.4 | 483.9 (23.6%) |
| 0.1   | 73.5 | 1709.5 | 424.0 (20.7%) |

_(update this table with the wider alpha sweep -- 0.1 to 10 -- once that finishes running)_

### Qualitative feature check

Some features already show clean, single-concept behavior. Feature 2 (at alpha = 0.01) fires almost exclusively on bibliographic citation text:

```
12.66 | National Mission; Society for the Preservation of Christian Knowledge, 1916
12.61 | A New Epiphany; Society for the Preservation of Christian Knowledge, 1919
12.54 | Guardian Angel; Society for the Preservation of Christian Knowledge, 1923
```

Other features (0 and 1) are still loosely thematic rather than cleanly monosemantic, consistent with the sparsity penalty not yet being strong enough at this alpha.

## Known issues / next steps

- Sparsity penalty needs further tuning — testing a wider alpha range (0.1–10) with longer training (80 epochs) to find a value that meaningfully reduces the number of active features per token.
- Once a working alpha is found, scale from the 500-activation sanity subset up to several thousand activations for a real training run.
- Compare against a PCA baseline on the same activations, to demonstrate (as the paper does) that sparse coding finds cleaner features than a simple linear decomposition.
- Consider a lightweight automated interpretability check (even a smaller open model instead of GPT-4) if time allows, to move beyond manual inspection.

## How to run

See `sae_reproduction_starter.ipynb`. Open in Google Colab, set runtime to T4 GPU, Runtime → Run all. Dependencies are pinned in `requirements.txt`.

## Provenance

See `PROVENANCE.md`. In short: the full notebook is written by us, implementing equations and default hyperparameters taken directly from the paper. No code was copied or adapted from the official repository.
