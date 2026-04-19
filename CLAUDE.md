# CLAUDE.md — RNN & LSTM for Sequential Text Generation

## Project Summary

Pure-NumPy implementation of Vanilla RNN and LSTM language models trained on an Alice in Wonderland excerpt to generate word-by-word text continuations. The entire forward pass, backward pass (BPTT), and Adam optimizer are hand-coded without any deep-learning framework. The project demonstrates that sequence models can be built from first principles while still reaching ~90% word-prediction accuracy on a small corpus.

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook "RNN and LSTM Network .ipynb"
```

Execute cells in order. The main training cell runs 10,800 iterations (50 epochs, batch size 10, ~216 batches/epoch). On a laptop CPU this takes 2–5 minutes.

Key parameters in the training cell:

| Variable | Meaning | Default |
|---|---|---|
| `D` | Word embedding dimension | 10 |
| `H` | Hidden state dimension | 35 |
| `T` | Timesteps per sequence | 15 |
| `N` | Batch size | 10 |
| `max_epoch` | Training epochs | 50 |
| `cell_type` | `'rnn'` or `'lstm'` | `'rnn'` |

Change `cell_type='lstm'` to compare LSTM performance.

Text generation is in the final cell. `text_length` controls how many words to generate. The seed word defaults to `'she'` but any word in the corpus vocabulary works.

---

## Model Architecture Details

### LanguageModelRNN

Three sequential components stored in `lib/rnn.py`:

1. **Preprocess** — embedding lookup: maps integer word index → dense vector of dim `D` using a learned weight matrix `(vocab_size × D)`.
2. **RNN / LSTM cell** — processes the embedded sequence `(N, T, D)` → hidden states `(N, T, H)`.
3. **Postprocess** — linear projection `(H → vocab_size)` produces logit scores for next-word prediction.

### Vanilla RNN cell (`VanillaRNN` in `lib/layer_utils.py`)

```
h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)
```

Single weight matrix for input (`W_x`), single recurrent matrix (`W_h`), bias `b`. Suffers from vanishing gradients for long sequences.

### LSTM cell (`LSTM` in `lib/layer_utils.py`)

Four gating operations (input, forget, output, cell gate) using the concatenated `[x_t, h_{t-1}]` vector. Cell state `c_t` acts as a long-term memory highway, alleviating vanishing gradients.

### Loss

Temporal Softmax Cross-Entropy (`temporal_softmax_CE_loss`) — computes cross-entropy at every timestep and averages across the sequence, ignoring padded positions via a mask.

### Optimizer

Adam with `lr = 5e-4`, `beta1 = 0.9`, `beta2 = 0.999`, `epsilon = 1e-8`. SGD and SGD+Momentum implementations also exist in `lib/optim.py`.

### Data

- Corpus: `data/alice.txt` — 2,170 words, 778 unique tokens (word-level, lowercased).
- Tokenisation: `re.split(' |\n', text.lower())` — no stemming or special token handling.

---

## Current Limitations

1. **Tiny corpus** — 778-word vocabulary from a single short passage. The model memorises rather than generalises; it cannot produce novel vocabulary.
2. **No temperature sampling** — the `model.sample()` method uses argmax (greedy decoding). Output can be repetitive for some seed words.
3. **Deprecated NumPy API** — `np.int` is used in Cell 4 (removed in NumPy 1.24+). Replace with `np.int64` or `int` to run on modern environments.
4. **No model persistence** — weights are not saved to disk; every run retrains from scratch.
5. **Word-level only** — cannot generate words not seen during training (closed vocabulary).
6. **No batched inference** — `model.sample()` is single-example, CPU-only.
7. **No padding / variable-length support** — all sequences must be exactly `T` tokens.

---

## Enhancement TODO List (Portfolio Demo-ability)

### Quick wins

- [ ] Fix `np.int` deprecation warning (one-line change, ensures compatibility with NumPy 1.24+).
- [ ] Add `model.save()` / `model.load()` using `np.savez` so weights persist between sessions.
- [ ] Add temperature parameter to `model.sample()` — divide logits by `temperature` before softmax. Temperature = 1.0 is unchanged; < 1 is more deterministic; > 1 is more creative.
- [ ] Write a simple CLI: `python generate.py --seed alice --length 50 --temperature 0.8`.

### Medium lift

- [ ] **Gradio UI** — a 20-line Gradio app with:
  - Text input for seed word
  - Slider for temperature (0.5 – 2.0)
  - Slider for output length (10 – 100 words)
  - Radio button to toggle RNN vs LSTM
  - Typewriter-style output using `gr.Textbox` with streaming via `yield`
- [ ] **Hugging Face Space** — push the Gradio app as a free CPU Space. Pre-train weights, serialize with `np.savez`, load at startup. No GPU needed.
- [ ] Train on a larger Alice corpus (full book ~26 000 words) for more coherent output.
- [ ] Character-level mode: swap word tokenisation for character tokenisation; smaller vocab, more generalisable.

### Big lift

- [ ] Port the LSTM to PyTorch (or JAX) to enable GPU acceleration and larger hidden dims.
- [ ] Add beam-search decoding for higher-quality text.
- [ ] Train on a themed corpus (e.g. Shakespeare, news headlines) and let users choose the domain.
- [ ] Visualise LSTM gate activations in the Gradio UI to show what the model "attends to".

---

## Recommended Demo Tier

**Medium lift — Gradio UI on Hugging Face Space.**

A Gradio interface with temperature slider and RNN-vs-LSTM toggle makes the core learning (gating helps long-range coherence) immediately tangible to any reviewer without requiring them to run a notebook locally. Deploying to a Hugging Face Space takes the project from "a notebook I ran once" to "a live URL I can share in a job application or interview", which is the single highest-leverage portfolio upgrade for the effort involved (estimated 2–4 hours of work).
