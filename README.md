# GPT from Scratch and Sentiment Analysis

Two NLP builds in PyTorch: a character-level GPT that learns to write Shakespeare-style text, and a sentiment classifier wrapped in a small chatbot.

> Lab 1 of *Advanced Deep Learning (D7047E)*, Luleå University of Technology, spring 2024. Group 7; see [Team](#team).

## 1. Character-level GPT

A decoder-only transformer trained on Tiny Shakespeare (about 1.1 million characters, 65 unique), built up step by step:

1. Character tokeniser and a bigram baseline
2. Single-head, then **multi-head causal self-attention**
3. Feed-forward layers, **residual connections** and layer normalisation, stacked into transformer blocks
4. Training with AdamW, tracking train and validation loss

Final model: 6 transformer layers, 6 attention heads, 256-dimensional embeddings, a 128-character context window and 20% dropout, trained for 5,000 steps at a learning rate of 3e-4.

**Result:** validation loss fell from 4.58 (random) to **1.53**. Sample output, generated character by character:

```
MENENIUS:
Mercy, one dreams; why, counsel the hopesition.

CORIOLANUS:
Yet be bankle, thy heaven worse,
Your twos are me, you comed's tongue rosen:
Hosteff, each your soul means with our poorsue.
```

The model has learned the format (speaker names, line breaks, verse rhythm) and plenty of real words, but not yet meaning, which is what you expect from a small model trained for a few thousand steps. The implementation follows Andrej Karpathy's well-known "Let's build GPT" walkthrough.

## 2. Sentiment analysis chatbot

A feed-forward neural network trained on the [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) Amazon product reviews (included in [`data/`](data)), with NLTK tokenisation and stop-word removal, and TF-IDF features. It is wrapped in a simple chat loop:

```
You:      This does not work!
Chatbot:  It seems like you're expressing a negative sentiment.
You:      Im so happy with the support i have got from you!
Chatbot:  It seems like you're expressing a positive sentiment.
```

## Repository contents

| Path | Contents |
|---|---|
| [`notebooks/1.2_char_level_gpt.ipynb`](notebooks/1.2_char_level_gpt.ipynb) | Transformer language model from scratch |
| [`notebooks/1.1_sentiment_analysis.ipynb`](notebooks/1.1_sentiment_analysis.ipynb) | Sentiment classifier and chatbot |
| [`data/amazon_cells_labelled.txt`](data/amazon_cells_labelled.txt) | Labelled Amazon reviews |
| [`docs/`](docs) | Lab report and theory answers |

```bash
pip install torch nltk scikit-learn pandas matplotlib jupyter
# The GPT notebook expects Tiny Shakespeare as input.txt:
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
jupyter notebook notebooks/
```

## Team

Group 7, D7047E Advanced Deep Learning: **Aamer Barood** and teammates. The original group repository, with full commit history, is [Gabmyr/D7047E-Labs-Group-7](https://github.com/Gabmyr/D7047E-Labs-Group-7).
