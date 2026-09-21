# Gen-Trans

Gen-Trans is a Bulgarian-to-English neural machine translation system built
from scratch in PyTorch. It uses a decoder-only Transformer language model to
generate an English translation after a Bulgarian source sequence.

This repository contains my coursework in **Information Search and Retrieval:
Application of Deep Learning**, taught by **Prof. Stoyan Mihov** during the
**Winter 2025/2026 semester** at Sofia University's Faculty of Mathematics and
Informatics.

## Highlights

- Custom byte-pair encoding (BPE) tokenizer with 30,000 merge operations
- Decoder-only Transformer with masked multi-head self-attention
- Three Transformer layers, eight attention heads, and 512-dimensional hidden
  states
- KV caching for efficient autoregressive generation
- Greedy decoding and length-normalized beam search
- Label smoothing, gradient clipping, linear warmup, and cosine learning-rate
  decay
- Reproducible training with tracked loss, gradient norm, learning rate, and
  validation perplexity

The model is trained on the included parallel corpus of 180,000 sentence
pairs, with 1,000 validation pairs and 6,000 test pairs. It does not use
pretrained models or additional training corpora.

## Results

| Decoding strategy | BLEU-4 | Test perplexity |
| --- | ---: | ---: |
| Greedy | 41.08 | 12.26 |
| Beam search (width 4) | **42.57** | **12.26** |

Beam search improves BLEU-4 by 1.49 points over greedy decoding. See the
[project report](doc.pdf) for the architecture, experiments, training curves,
and references.

## Setup

Create the Conda environment defined by the project:

```bash
conda env create -f tii.yml
conda activate tii
```

The default device is `cuda:0`. To run on CPU, change `device` in
`parameters.py` to `torch.device("cpu")`.

## Usage

Prepare the tokenized training and validation data:

```bash
python run.py prepare
```

Train a model, or continue from the saved model and optimizer state:

```bash
python run.py train
python run.py extratrain
```

Evaluate the included checkpoint on the test split:

```bash
python run.py perplexity en_bg_data/test.bg en_bg_data/test.en
python run.py translate en_bg_data/test.bg predictions.en
python run.py bleu en_bg_data/test.en predictions.en
```

Generate a continuation from an arbitrary prefix. Include the special tokens
that mark the sequence start and the beginning of the translation:

```bash
python run.py generate "<S> Това е пример . <TRANS>"
```

## Repository structure

- `model.py` - Transformer, attention, KV cache, and decoding strategies
- `bpe_tokenization.py` - BPE learning, encoding, and decoding
- `run.py` - data preparation, training, translation, generation, and evaluation
- `parameters.py` - model and training configuration
- `training_metrics.json` and `plots/` - recorded training diagnostics
- `doc.pdf` - full technical report
