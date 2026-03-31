# LLM Governor Experiment

**Can Large Language Models Govern a Multi-Objective Economy?**

This repository contains the experimental code, agent-based model, and analysis pipeline for the paper submitted to *Economic Theory*.

## Overview

This experiment tests whether an open-source LLM (Qwen3-8B) can function as a macroeconomic governor within a multi-agent economic simulation. We compare three governance strategies across two economic cycles using a 3×2 factorial design:

| Strategy | Description |
|---|---|
| **Thinking Mode** | Qwen3-8B with extended chain-of-thought reasoning (`/think` tag) |
| **Non-Thinking Mode** | Qwen3-8B with direct response generation (`/no_think` tag) |
| **Rule-Based Heuristic** | Taylor-rule-inspired deterministic baseline |

The agent-based model features three interconnected currency zones (EUR, USD, YEN) with 12 policy levers targeting the Eurozone Price Index and Credit Utilization.

## Repository Structure

```
.
├── Experiment_Script.py      # Full experiment: ABM + vLLM inference + analysis
├── results/                  # Output data (generated after running)
│   ├── final_results_4096.csv
│   ├── table1_descriptive_4096.csv
│   ├── history_*.json        # Per-replication economic trajectories
│   ├── levers_*.json         # Per-replication governance decisions
│   └── fig*.png              # Figures 1-5
└── README.md
```

## Usage

### Google Colab (recommended)

1. Open a new Colab notebook with an **L4 GPU** runtime
2. Paste the contents of `Experiment_Script.py` into a single cell
3. Run the cell — the script handles all installation, model loading, experimentation, and analysis

### Local execution

```bash
pip install vllm scipy statsmodels matplotlib seaborn tqdm
python Experiment_Script.py
```

Requires a local GPU with ≥24GB VRAM and CUDA support.

## Experimental Design

- **Conditions**: 3 strategies × 2 economic cycles = 6 conditions
- **Replications**: N = 3 per condition (18 total runs)
- **Simulation**: 120 weeks per replication, 8-week calibration burn-in
- **Decision interval**: Every 8 weeks (14 decision points per replication)
- **Targets**: Price Index ∈ [14.0, 30.0], Credit Utilization ∈ [0.74, 0.86]
- **Inference**: Temperature = 0.0 (greedy decoding), max 4,096 tokens

## Key Findings

1. **Ineffective deliberation**: Thinking mode consumes ~32,000 tokens per replication but fails to move the economy toward targets
2. **Counterproductive action**: Non-thinking mode frequently pushes the economy *away* from targets
3. **Rule-based superiority**: The Taylor-rule-inspired heuristic significantly outperforms both LLM strategies (Kruskal-Wallis H = 14.01, p = .016 for PI distance from target)

## Model

The LLM used is [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) via the vLLM inference engine. Model weights are publicly available on Hugging Face.

## Citation

[To be added after publication]

## License

This project is licensed under the MIT License.
