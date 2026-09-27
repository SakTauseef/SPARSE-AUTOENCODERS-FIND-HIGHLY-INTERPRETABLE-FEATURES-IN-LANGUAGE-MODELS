# Sparse Autoencoder Reproduction

Reproduction of: Cunningham, Ewart, Riggs, Huben & Sharkey (2023), *"Sparse Autoencoders Find Highly Interpretable Features in Language Models"*, arXiv:2309.08600.

Official reference repo (used for reference only, not copied): https://github.com/HoagyC/sparse_coding

## What the paper does

Neurons inside language models are often *polysemantic*  a single neuron reacts to several unrelated concepts, which makes the model hard to interpret. The paper's hypothesis is that this happens because of *superposition*: the model represents more concepts than it has neurons, so it packs multiple concepts into overlapping directions, relying on those concepts rarely firing at the same time.

The proposed fix: freeze the trained language model, record its internal activations, and train a separate **sparse autoencoder** on those activations. The autoencoder learns a larger set of "dictionary features" that, empirically, tend to be far more interpretable each one closer to representing a single clean concept — than the original neurons.

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

### Sparsity: finding the right alpha

Our first sweep (alpha 0.001–0.1) showed almost no effect — sparsity barely moved even at 100x range. A wider sweep (alpha 0.1–10, 80 epochs each) revealed the penalty only starts biting past ~0.3:

| alpha | final recon | final sparsity | avg active features / token |
|---|---|---|---|
| 0.1  |  21.16 | 1390.65 | 335.4 (16.4%) |
| 0.3  |  60.53 |  801.21 | 140.6 (6.9%) |
| **1.0**  | **174.19** |  **357.89** | **33.3 (1.6%)** |
| 3.0  | 679.97 |  174.61 | 6.6 (0.3%) |
| 10.0 | 870.90 |   88.97 | 1.6 (0.1%) |

**Chosen value: alpha = 1.0.** This lands in the target range (roughly 10-50 active features per token, i.e. genuinely sparse rather than "everything a little bit on"), while alpha = 3.0 and 10.0 overcorrect -- too few features stay active, reconstruction quality collapses (recon rises sharply), suggesting the model is starved of the capacity it needs. alpha = 0.1 and 0.3 remain too permissive, with hundreds of features active per token.

### Qualitative feature check (alpha = 1.0)

Feature 2 shows real, coherent specialization -- 4 of its top 5 activating texts are specifically about the Little Rock Arsenal during the Civil War:

```
1.01 | Lt. Col. Dunnington continued to build up his works at Little Rock until November 1862...
0.93 | This ammunition, and that which I brought with me, was rapidly prepared for use...
0.85 | Lt. Col. Dunnington's "Returns for the month of August, 1862, at Little Rock Arsenal..."
0.83 | The Confederate ordnance establishment at Little Rock was reactivated in August, 1862...
```

Features 0 and 1 are still loosely mixed across topics (a video game, fish biology, art history) rather than one clean concept each. With only 200 documents (500 sampled activations) used for the sanity-check runs so far, there likely isn't enough data diversity yet for all 2048 features to specialize -- this is the next thing to test once training scales up.

### Scaled-up run (401,882 activations)

Trained `full_sae` on the complete activation set from 5000 documents (up from 500 in the sanity check), at the tuned alpha = 1.0, for 150 epochs. Loss converged and flattened by roughly epoch 20-30, meaning fewer epochs would likely suffice for future runs.

**Sparsity held consistently at scale:** 33.1 / 2048 active features per token (1.62%) -- nearly identical to the 33.3 (1.6%) seen on the 500-sample sanity check, confirming alpha = 1.0's sparsity behavior isn't an artifact of a small sample.

**Dead features:** 159 / 2048 (7.8%) never fire across a 2000-sample check. This matches a known limitation the paper itself reports (Section 5) -- not every dictionary slot ends up used, particularly outside the residual stream.

*Note on raw loss magnitude:* the reported loss/recon/sparsity numbers here are much larger than the sanity-check run's because the training loop sums loss across every batch in an epoch, and this run has far more batches (~6,280 vs ~8). Per-batch cost is comparable, not worse.

### Qualitative feature check at scale

With more data, several features are now cleanly monosemantic:

- **Feature 1** fires almost exclusively on text about the *Hellblazer* / *Constantine* comic and film franchise -- a genuinely narrow, single concept.
- **Feature 2** consistently activates on Holocaust historiography and denial discourse.
- **Feature 0** leans toward physical architecture and sculpted structures (churches, caves, temple panels).
- Features 3 and 4 remain more mixed, showing specialization is uneven across the dictionary -- consistent with the paper's own finding that not every feature reaches the same interpretability quality.

Model checkpoint (`sae_checkpoint_alpha1.0.pt`) and full per-epoch loss history (`full_training_history.csv`) are saved in this repo as evidence for this run.

## Known issues / next steps

- Alpha for the sparsity penalty is tuned and confirmed stable at scale (alpha = 1.0, ~1.6% of features active per token, both at 500 and 401,882 activations).
- Compare against a PCA baseline on the same activations, to demonstrate (as the paper does) that sparse coding finds cleaner features than a simple linear decomposition.
- Consider a lightweight automated interpretability check (even a smaller open model instead of GPT-4) if time allows, to move beyond manual inspection.
- Planned Stage 4 experiment: data-size scaling -- compare feature coherence and dead-feature rate across a range of dataset sizes (500 -> 20,000 -> 401,882 activations), since the two runs so far already suggest larger data improves specialization.

## How to run

See `sae_reproduction_starter.ipynb`. Open in Google Colab, set runtime to T4 GPU, Runtime → Run all. Dependencies are pinned in `requirements.txt`. Running section 7 onward reproduces the full-scale training run; `sae_checkpoint_alpha1.0.pt` (trained weights) and `full_training_history.csv` (per-epoch loss log) are the artifacts that run produces.

## Provenance

See `PROVENANCE.md`. In short: the full notebook is written by us, implementing equations and default hyperparameters taken directly from the paper. No code was copied or adapted from the official repository.
