"""SAE Feature Explorer — Milestone 2 deployment.

Type a sentence. The app runs it through frozen Pythia-70M, takes the residual
stream at layer 3, passes it through our trained sparse autoencoder, and shows
which learned feature fires most on each token, with real examples of where
that feature fires in the WikiText training data.
"""
import html
import json

import streamlit as st
import torch
import torch.nn.functional as F

CKPT_PATH = "sae_final.pt"
EXAMPLES_PATH = "feature_examples.json"
DEFAULT_TEXT = ("The film was directed by John Ross and released in 1954. Shortly after, "
                "a tropical storm hit Galveston, and the Football League season was postponed until March.")

st.set_page_config(page_title="SAE Feature Explorer", page_icon="🔍", layout="wide")


# ----------------------------------------------------------------- loading
class SparseAutoencoder(torch.nn.Module):
    """Same architecture as the training notebook: c = ReLU(Mx + b), x_hat = M^T c."""
    def __init__(self, d_model, dict_size):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros(dict_size, d_model))
        self.b_enc = torch.nn.Parameter(torch.zeros(dict_size))

    def encode(self, x):
        return F.relu(x @ self.W.T + self.b_enc)


@st.cache_resource(show_spinner="Loading Pythia-70M and the sparse autoencoder…")
def load_resources():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).eval()
    sae = SparseAutoencoder(cfg["d_model"], cfg["dict_size"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return tokenizer, model, sae, ckpt


@st.cache_data
def load_examples():
    with open(EXAMPLES_PATH) as f:
        return json.load(f)


@torch.no_grad()
def analyze(text, tokenizer, model, sae, ckpt):
    cfg, norm = ckpt["config"], ckpt["norm"]
    ids = tokenizer(text, truncation=True, max_length=cfg["max_length"])["input_ids"]
    if len(ids) < 2:
        return None
    hidden = model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states[cfg["layer"]][0]
    skip = set(cfg.get("skip_token_ids", []))
    tokens = [tokenizer.decode([i]) for i in ids]
    feats = torch.zeros(len(ids), cfg["dict_size"])
    analyzed = [p for p in range(len(ids)) if p > 0 and ids[p] not in skip]   # same rule as training
    if analyzed:
        x = (hidden[analyzed].float() - norm["mean"]) / norm["scale"]
        feats[analyzed] = sae.encode(x)
    return tokens, feats, analyzed


# ----------------------------------------------------------------- rendering helpers
def show_tok(t):
    return html.escape(t.replace("\n", "⏎")).replace(" ", "&nbsp;") or "&nbsp;"


def render_tokens(tokens, feats, analyzed, selected):
    top_val, top_idx = feats.max(dim=1)
    peak = max(top_val.max().item(), 1e-6)
    spans = []
    for p, t in enumerate(tokens):
        if p in analyzed and top_val[p] > 0:
            a = 0.12 + 0.75 * (top_val[p].item() / peak)
            tip = f"strongest feature #{top_idx[p].item()} (activation {top_val[p].item():.2f})"
            style = f"background: rgba(255,140,0,{a:.2f});"
        else:
            tip = "not analyzed (first token / line break)" if p not in analyzed else "no feature active"
            style = "background: transparent; opacity: 0.55;"
        border = "outline: 2px solid #1f77b4;" if p == selected else ""
        spans.append(f'<span title="{html.escape(tip)}" style="{style}{border}padding:3px 1px;'
                     f'border-radius:4px;margin:1px;line-height:2.2;">{show_tok(t)}</span>')
    st.markdown(f'<div style="font-family:monospace;font-size:1.05rem;">{"".join(spans)}</div>',
                unsafe_allow_html=True)
    st.caption("Darker orange = the token's strongest feature fires harder. Hover a token for its top feature. "
               "Faded tokens (the first token and line breaks) are skipped, exactly as in training.")


def render_examples(examples, feature, k=8):
    ex = examples.get(str(feature), [])
    if not ex:
        st.write("No stored examples for this feature (it never fired in the scanned training data).")
        return
    rows = []
    for e in ex[:k]:
        rows.append(f'<div style="font-family:monospace;font-size:0.9rem;margin:2px 0;">'
                    f'<span style="color:gray;">{e["act"]:.2f}&nbsp;|&nbsp;</span>'
                    f'{show_tok(e["left"])}<b style="background:rgba(255,140,0,0.75);padding:0 2px;">'
                    f'{show_tok(e["token"])}</b>{show_tok(e["right"])}</div>')
    st.markdown("".join(rows), unsafe_allow_html=True)


# ----------------------------------------------------------------- page
tokenizer, model, sae, ckpt = load_resources()
examples = load_examples()
cfg, metrics = ckpt["config"], ckpt.get("metrics", {})

with st.sidebar:
    st.header("About this model")
    st.markdown(
        f"- **Language model:** `{cfg['model_name']}` (frozen)\n"
        f"- **Layer:** residual stream after layer {cfg['layer']}\n"
        f"- **SAE features:** {cfg['dict_size']:,}\n"
        f"- **L1 coefficient α:** {cfg['alpha']:g}")
    if metrics:
        st.markdown("**Held-out evaluation**")
        c1, c2 = st.columns(2)
        c1.metric("Active features / token", f"{metrics['l0']:.1f}")
        c2.metric("LM loss recovered", f"{100 * metrics['loss_recovered']:.0f}%")
    st.markdown("---")
    st.caption("Reproduction of Cunningham et al. (ICLR 2024), *Sparse Autoencoders Find Highly "
               "Interpretable Features in Language Models*. Example contexts come from WikiText-103.")

st.title("🔍 Sparse Autoencoder Feature Explorer")
st.write("Type a sentence. The app runs it through Pythia-70M, decomposes the layer-3 activations with our "
         "sparse autoencoder, and shows which learned feature each token activates — plus real training "
         "text where that same feature fires, so you can judge what concept it represents.")

tab_text, tab_feature = st.tabs(["Analyze text", "Look up a feature"])

with tab_text:
    text = st.text_area("Your text", DEFAULT_TEXT, height=100, max_chars=600)
    result = analyze(text, tokenizer, model, sae, ckpt) if text.strip() else None
    if result is None:
        st.info("Enter at least two tokens of text.")
    else:
        tokens, feats, analyzed = result
        if not analyzed:
            st.info("Nothing to analyze in this text.")
        else:
            options = [f"{p}: {tokens[p].strip() or repr(tokens[p])}" for p in analyzed]
            default = int(feats[analyzed].max(dim=1).values.argmax())
            picked = st.selectbox("Pick a token to inspect (default: the token with the strongest feature)",
                                  options, index=default)
            choice = analyzed[options.index(picked)] if picked in options else analyzed[default]
            render_tokens(tokens, feats, analyzed, choice)

            n_active = int((feats[choice] > 0).sum())
            st.subheader(f"Token “{tokens[choice].strip()}” — {n_active} of {cfg['dict_size']:,} features active")
            vals, idxs = feats[choice].topk(5)
            shown = [(i.item(), v.item()) for i, v in zip(idxs, vals) if v > 0]
            if not shown:
                st.write("No feature fires on this token.")
            for rank, (f, v) in enumerate(shown):
                with st.expander(f"Feature #{f} — activation {v:.2f}", expanded=(rank == 0)):
                    st.caption("Where this feature fires most strongly in the training data:")
                    render_examples(examples, f)

with tab_feature:
    st.write("Enter any feature number to see the training contexts where it fires most strongly.")
    live = sorted(int(k) for k in examples)
    f = st.number_input("Feature number", min_value=0, max_value=cfg["dict_size"] - 1,
                        value=live[0] if live else 0, step=1)
    st.caption(f"{len(live):,} of {cfg['dict_size']:,} features have stored examples.")
    render_examples(examples, int(f), k=10)
