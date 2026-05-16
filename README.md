# TriEval: A Resource-Efficient Pipeline for LLM Bias, Toxicity, and Truthfulness Assessment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/physics-vibes15/TriEval/blob/main/TriEval_Colab_HF.ipynb)

**Authors:** Akshatha Srikantha, Yash Jajoo, Shyamal Lakhanpal, Manpreet Singh

---

## Overview

The market for LLM evaluation tools is growing, but most require expensive hardware, cloud infrastructure, or only test one dimension at a time. TriEval was built to fix that.

TriEval is an open-source pipeline that evaluates Large Language Models (LLMs) across three safety dimensions simultaneously:

- **Toxicity** — Does the model refuse to generate harmful content?
- **Truthfulness** — Does the model answer factual questions correctly?
- **Bias** — Does the model treat different demographic groups equally?

TriEval runs entirely on **Google Colab's free tier** using the **HuggingFace Inference API** for open-source models and the **Anthropic API** for Claude Haiku. Total cost for all three experiments is under **$2 USD**.

---

## Models Evaluated

| Model | Type | Parameters | How it runs |
|---|---|---|---|
| Meta Llama 3 8B | Open-source | 8B | HuggingFace Inference API (free) |
| Mistral AI Mistral 7B | Open-source | 7B | HuggingFace Inference API (free) |
| Google Gemma 2 9B | Open-source | 9B | HuggingFace Inference API (free) |
| Anthropic Claude Haiku | Closed-source | N/A | Anthropic API (~$2 total) |

---

## Key Results

| Model | Toxicity ↓ | Truthfulness ↑ | Bias Rate ↓ |
|---|---|---|---|
| Gemma 2 9B | 0.053 | 81.0% | 0.0% |
| Mistral 7B | 0.077 | 55.0% | 0.0% |
| Llama 3 8B | 0.053 | 41.0% | 0.0% |
| Claude Haiku | 0.100 | 0.0%* | 0.0% |

*Claude Haiku scored 0% on truthfulness due to a format compliance issue, not a knowledge deficiency. See the paper for full explanation.

---

## Experiments

### Experiment 1 — Toxicity (30 prompts)
- 30 adversarial prompts designed by the authors
- Inspired by ToxiGen adversarial prompt design principles
- 6 harm categories: personal insults, body shaming, threats, hate speech, age discrimination, online bullying
- Scored 0.0 to 1.0 by Claude Haiku as judge-LLM

### Experiment 2 — Truthfulness (60 questions)
- 60 questions from the TruthfulQA dataset (HuggingFace, free)
- Multiple choice format — models answer with a single letter
- Ground truth verified against published TruthfulQA labels
- Pearson correlation of 0.935 with published TruthfulQA MC1 scores

### Experiment 3 — Bias (30 paired prompt sets)
- 30 paired prompts designed by the authors
- Inspired by BiasAsker paired-prompt methodology
- 5 demographic dimensions: gender, race, religion, age, nationality
- 6 pairs per dimension
- Scored by Claude Haiku comparing both responses for differential treatment

---

## How to Run

### Option 1 — Google Colab (Recommended)

Click the button below to open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/physics-vibes15/TriEval/blob/main/TriEval_Colab_HF.ipynb)

**Before running:**

1. Get a free HuggingFace token at [huggingface.co](https://huggingface.co) → Settings → Access Tokens → New Token (Read)
2. Get an Anthropic API key at [console.anthropic.com](https://console.anthropic.com)
3. In Colab click the **key icon** on the left sidebar and add:
   - `HF_TOKEN` → your HuggingFace token
   - `ANTHROPIC_API_KEY` → your Anthropic key
4. Run all cells from top to bottom

No GPU required. No local installation needed. Results download automatically as CSV files.

### Option 2 — Local (Mac/Linux)

```bash
# Clone the repository
git clone https://github.com/physics-vibes15/TriEval.git
cd TriEval

# Install dependencies
pip install anthropic datasets pandas numpy requests huggingface_hub

# Set your API keys
export ANTHROPIC_API_KEY=your-anthropic-key
export HF_TOKEN=your-huggingface-token

# Run the notebook
jupyter notebook TriEval_Colab_HF.ipynb
```

---

## Output Files

After running all three experiments, four CSV files are generated:

| File | Contents |
|---|---|
| `results_toxicity.csv` | Per-model toxicity scores for all 30 prompts |
| `results_truthfulness.csv` | Per-question accuracy results for all 60 questions |
| `results_bias.csv` | Per-pair bias detection results for all 30 pairs |
| `results_summary.csv` | Final averages across all three dimensions |

---

## Requirements

- Python 3.8+
- Anthropic API key (for Claude Haiku)
- HuggingFace token (free, no credit card required)
- Google Colab free tier OR local Jupyter environment

### Python Dependencies

```
anthropic
datasets
pandas
numpy
requests
huggingface_hub
```

---

## Repository Structure

```
TriEval/
├── TriEval_Colab_HF.ipynb    # Main experiment notebook (run this)
├── results_toxicity.csv       # Toxicity experiment results
├── results_truthfulness.csv   # Truthfulness experiment results
├── results_bias.csv           # Bias experiment results
├── results_summary.csv        # Summary of all results
└── README.md                  # This file
```

---

## Citation

If you use TriEval in your research, please cite:

```
Srikantha, A., Jajoo, Y., Lakhanpal, S., & Singh, M. (2025).
TriEval: A Resource-Efficient Pipeline for LLM Bias, Toxicity,
and Truthfulness Assessment.
https://github.com/physics-vibes15/TriEval
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright © 2025 Akshatha Srikantha. All rights reserved.

---

## Acknowledgments

Thanks to the research teams behind TruthfulQA, ToxiGen, BiasAsker, and the LLM-as-judge methodology whose foundational work made this evaluation framework possible. Thanks also to the HuggingFace and Anthropic teams for providing accessible APIs that make this kind of research possible without large compute budgets.
