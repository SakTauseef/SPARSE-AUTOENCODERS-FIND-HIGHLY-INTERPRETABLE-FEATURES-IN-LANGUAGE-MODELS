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

## Results

All numbers in the README (loss curves, alpha sweep table, feature examples) are from our own training runs, not the paper's reported results.
