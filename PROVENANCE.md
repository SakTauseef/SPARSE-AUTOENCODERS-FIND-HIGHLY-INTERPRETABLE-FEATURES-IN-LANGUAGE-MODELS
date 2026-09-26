# Provenance

Tracks the origin of every major component, per the assignment's academic integrity requirements.

| Component | Status | Notes |
|---|---|---|
| `SparseAutoencoder` class (encoder/decoder) | Written by us | Implements equations from paper Section 2 (`c = ReLU(Mx+b)`, `x̂ = Mᵀc`) |
| `sae_loss` function | Written by us | Implements paper's loss formula (`L = \|\|x-x̂\|\|² + α\|\|c\|\|₁`) |
| Activation collection (`collect_activations`) | Written by us | Uses `output_hidden_states=True`, our own approach; paper's official repo uses manual hooks instead |
| Training loop (`train_sae`) | Written by us | Standard Adam optimizer training loop with weight re-normalization each step, per paper Appendix B |
| Default hyperparameters (alpha ≈ 8e-4, R = 4) | Referenced from paper | Starting values taken from the paper's reported settings, then tuned ourselves via sweep |
| Dataset (WikiText-103) | Substituted | Paper uses the Pile; we substitute WikiText-103 for accessibility (see README for implications) |
| Model (Pythia-70M) | Reused, unmodified | Pretrained weights loaded as-is from Hugging Face (`EleutherAI/pythia-70m`), never fine-tuned or altered |
| Official repo (`HoagyC/sparse_coding`) | Referenced, not copied | Consulted for understanding the paper's intended architecture; no code copied or adapted into this repository |
| Scale-up training loop (`full_sae`, section 7) | Written by us | Same `train_sae` function reused as-is on the full 401,882-activation set; no new code, just a larger run |
| Sparsity diagnostics (`active_features_per_token`, dead-feature check) | Written by us | Not present in the official repo's public tooling; written to answer our own debugging question about why sparsity plateaued |
| Checkpoint/history saving (section 7c) | Written by us | Standard `torch.save` / `csv` usage, no external reference |

## Change log

Update this section every time you commit code, not just at the end -- one line is enough.

- **[date]** Initial notebook: model loading, activation collection, autoencoder class, sanity check on one batch. All written by us from the paper's equations.
- **[date]** Ran first alpha sweep (0.001-0.1) on 500-activation sanity subset -- sparsity term wasn't decreasing meaningfully.
- **[date]** Wider alpha sweep (0.1-10, 80 epochs each) -- found alpha=1.0 gives ~1.6% active features/token, the target sparsity range.
- **[date]** Scaled up: 5000 documents -> 401,882 activations, trained `full_sae` at alpha=1.0 for 150 epochs. Sparsity held consistent with the sanity check; found several cleanly monosemantic features (e.g. Hellblazer/Constantine feature).

## Results

All numbers in the README (loss curves, alpha sweep table, feature examples) are from our own training runs, not the paper's reported results.
