# RNN & LSTM for Sequential Text Generation

> A from-scratch NumPy implementation of Vanilla RNN and LSTM that learns to generate Alice in Wonderland-style prose, word by word.

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sarvesh1karandikar/RNN-LSTM-for-sequential-text-generation/blob/main/RNN%20and%20LSTM%20Network%20.ipynb)

---

## What this does

This project trains a word-level language model on a short excerpt from *Alice in Wonderland* and then generates new text continuations from any seed word. Given the word `she`, for example, the trained model produces:

> *she was now only ten inches high, and she jumped up on to her feet in a moment: she found herself falling down a very deep well. either the well was very deep, or thought another moment down went...*

Both cell types — **Vanilla RNN** and **LSTM** — are implemented from scratch in pure NumPy (no PyTorch, no TensorFlow). Forward passes, backward passes, and gradient checks are all hand-rolled, making this an excellent reference for understanding how sequence models work under the hood.

---

## Architecture

| Hyperparameter | Value |
|---|---|
| Cell type | Vanilla RNN or LSTM (switchable) |
| Input embedding dim (`D`) | 10 |
| Hidden state dim (`H`) | 35 |
| Sequence / timestep length (`T`) | 15 words |
| Batch size (`N`) | 10 |
| Vocab size | 778 unique words |
| Corpus size | 2,170 words (Alice in Wonderland excerpt) |
| Training epochs | 50 |
| Optimizer | Adam (lr = 5e-4) |
| Loss function | Temporal Softmax Cross-Entropy |

The model pipeline is:

```
Input word index
    → Embedding layer (word_size → D)
    → RNN / LSTM cell (D → H)   [unrolled T timesteps]
    → Linear projection (H → vocab_size)
    → Softmax → next-word prediction
```

Gradient correctness is verified numerically before training (relative errors < 1e-7).

---

## Results

Training reaches **~90.5% word-prediction accuracy** after 50 epochs on the small Alice corpus.

| Epoch | Training Accuracy |
|---|---|
| 10 | 25.6 % |
| 20 | 56.3 % |
| 30 | 74.0 % |
| 40 | 84.3 % |
| 50 | **90.5 %** |

Loss drops from ~100 at iteration 1 to ~12.8 at iteration 10,500.

### Example generated text (seed word: `she`)

> *she was now only ten inches high, and she jumped up on to her feet in a moment: she found herself falling down a very deep well. either the well was very deep, or thought another moment down went*

*(The acceptable benchmark from the notebook)*

> *she was dozing off, and book-shelves; here and she tried to curtsey as she spoke--fancy curtseying as you're falling through the little door into a dreamy sort of way, 'do cats eat bats? do cats eat bats?' and sometimes,*

---

## Tech Stack

- **Python 3** — core language
- **NumPy** — all tensor math (no deep-learning framework)
- **Matplotlib** — loss / accuracy curves
- **Jupyter Notebook** — interactive experimentation
- `re` — text pre-processing

---

## Setup & Running

```bash
# 1. Clone the repo
git clone https://github.com/sarvesh1karandikar/RNN-LSTM-for-sequential-text-generation.git
cd RNN-LSTM-for-sequential-text-generation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook "RNN and LSTM Network .ipynb"
```

Run cells top-to-bottom. The training cell (~10,800 iterations) completes in a few minutes on a modern CPU.

To switch between cell types, change `cell_type='rnn'` to `cell_type='lstm'` in the `LanguageModelRNN(...)` constructor call.

---

## Open in Google Colab

Click the badge at the top, or use this direct link:

```
https://colab.research.google.com/github/sarvesh1karandikar/RNN-LSTM-for-sequential-text-generation/blob/main/RNN%20and%20LSTM%20Network%20.ipynb
```

No local install needed — Colab has NumPy and Matplotlib pre-installed.

---

## Live Demo concept

A polished portfolio demo would look like this:

1. User types a **seed word** (e.g. `alice`, `rabbit`, `curious`) into a text box.
2. A Gradio/Streamlit app calls `model.sample(seed_idx, length)`.
3. Words appear **one at a time** with a short delay — streaming typewriter effect.
4. A **temperature / creativity slider** (0.5 = conservative, 1.5 = wild) controls softmax sampling.
5. A toggle switches between the RNN and LSTM models live so the user can compare outputs.

This would be deployable as a **Hugging Face Space** (free tier, CPU-only) in under an hour.

---

## What I Learned / Key Insights

- **Vanishing gradients are real**: Vanilla RNN loses coherence after ~5 words; LSTM maintains context across the full 15-word window thanks to the cell state and gating mechanism.
- **Numerical gradient checking**: Implementing `eval_numerical_gradient` and verifying relative errors < 1e-7 gives high confidence that the backprop math is correct before spending GPU time on training.
- **Adam matters on small data**: SGD with momentum diverged on this tiny 778-word vocab; Adam converged smoothly to 90%+ accuracy.
- **Word-level vs character-level**: Word-level tokenization is faster to converge but has a hard vocabulary ceiling; character-level would generalise to new words.
- **Temperature sampling**: Greedy argmax produces repetitive loops; temperature-scaled softmax sampling produces more natural, varied text.
